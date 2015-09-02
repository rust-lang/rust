# Copyright 2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from __future__ import division, print_function
import json
import argparse
import sys
import re
import textwrap
import itertools

SPEC = re.compile(
    r'^(?:(?P<id>[iusfIUSF])(?:\((?P<start>\d+)-(?P<end>\d+)\)|'
    r'(?P<width>\d+)(:?/(?P<llvm_width>\d+))?)'
    r'|(?P<reference>\d+)(?P<modifiers>[vShdnwusDMC]*)(?P<force_width>x\d+)?)'
    r'(?:(?P<pointer>Pm|Pc)(?P<llvm_pointer>/.*)?)?$'
)

class PlatformInfo(object):
    def __init__(self, json):
        self._platform = json['platform']
        self._intrinsic_prefix = json['intrinsic_prefix']

    def intrinsic_prefix(self):
        return self._intrinsic_prefix

class IntrinsicSet(object):
    def __init__(self, platform, json):
        self._llvm_prefix = json['llvm_prefix']
        self._type_info = json['number_info']
        self._intrinsics = json['intrinsics']
        self._widths = json['width_info']
        self._platform = platform

    def intrinsics(self):
        for raw in self._intrinsics:
            yield GenericIntrinsic(self,
                                   raw['intrinsic'], raw['width'], raw['llvm'],
                                   raw['ret'], raw['args'])

    def platform(self):
        return self._platform

    def llvm_prefix(self):
        return self._llvm_prefix

    def width_info(self, bitwidth):
        return self._widths[str(bitwidth)]

    def number_type_info(self, value):
        data = self._type_info[value.__class__.__name__.lower()]
        bitwidth = value.bitwidth()
        def lookup(raw):
            if not isinstance(raw, dict):
                return raw

            try:
                return raw[str(bitwidth)]
            except KeyError:
                return raw['pattern'].format(bitwidth = bitwidth)

        return PlatformTypeInfo(value.llvm_name(),
                                {k: lookup(v) for k, v in data.items()})

class PlatformTypeInfo(object):
    def __init__(self, llvm_name, properties):
        self.properties = properties
        self.llvm_name = llvm_name

    def __getattr__(self, name):
        return self.properties[name]

    def vectorize(self, length, width_info):
        props = self.properties.copy()
        props.update(width_info)
        return PlatformTypeInfo('v{}{}'.format(length, self.llvm_name), props)

    def pointer(self):
        return PlatformTypeInfo('p0{}'.format(self.llvm_name), self.properties)

BITWIDTH_POINTER = '<pointer>'

class Type(object):
    def __init__(self, bitwidth):
        self._bitwidth = bitwidth

    def bitwidth(self):
        return self._bitwidth

    def modify(self, spec, width):
        raise NotImplementedError()

class Number(Type):
    def __init__(self, bitwidth):
        Type.__init__(self, bitwidth)

    def modify(self, spec, width):
        if spec == 'u':
            return Unsigned(self.bitwidth())
        elif spec == 's':
            return Signed(self.bitwidth())
        elif spec == 'w':
            return self.__class__(self.bitwidth() * 2)
        elif spec == 'n':
            return self.__class__(self.bitwidth() // 2)
        elif spec == 'v':
            return Vector(self, width // self.bitwidth())
        else:
            raise ValueError('unknown modification spec {}', spec)

    def type_info(self, platform_info):
        return platform_info.number_type_info(self)

class Signed(Number):
    def __init__(self, bitwidth, llvm_bitwidth = None):
        Number.__init__(self, bitwidth)
        self._llvm_bitwidth = llvm_bitwidth

    def compiler_ctor(self):
        if self._llvm_bitwidth is None:
            return 'i({})'.format(self.bitwidth())
        else:
            return 'i_({}, {})'.format(self.bitwidth(),
                                       self._llvm_bitwidth)

    def llvm_name(self):
        bw = self._llvm_bitwidth or self.bitwidth()
        return 'i{}'.format(bw)

    def rust_name(self):
        return 'i{}'.format(self.bitwidth())

class Unsigned(Number):
    def __init__(self, bitwidth, llvm_bitwidth = None):
        Number.__init__(self, bitwidth)
        self._llvm_bitwidth = llvm_bitwidth

    def compiler_ctor(self):
        if self._llvm_bitwidth is None:
            return 'u({})'.format(self.bitwidth())
        else:
            return 'u_({}, {})'.format(self.bitwidth(),
                                       self._llvm_bitwidth)

    def llvm_name(self):
        bw = self._llvm_bitwidth or self.bitwidth()
        return 'i{}'.format(bw)

    def rust_name(self):
        return 'u{}'.format(self.bitwidth())

class Float(Number):
    def __init__(self, bitwidth):
        assert bitwidth in (32, 64)
        Number.__init__(self, bitwidth)

    def compiler_ctor(self):
        return 'f({})'.format(self.bitwidth())

    def llvm_name(self):
        return 'f{}'.format(self.bitwidth())

    def rust_name(self):
        return 'f{}'.format(self.bitwidth())

class Vector(Type):
    def __init__(self, elem, length):
        assert isinstance(elem, Type) and not isinstance(elem, Vector)
        Type.__init__(self,
                      elem.bitwidth() * length)
        self._length = length
        self._elem = elem

    def modify(self, spec, width):
        if spec == 'h':
            return Vector(self._elem, self._length // 2)
        elif spec == 'd':
            return Vector(self._elem, self._length * 2)
        elif spec.startswith('x'):
            new_bitwidth = int(spec[1:])
            return Vector(self._elem, new_bitwidth // self._elem.bitwidth())
        else:
            return Vector(self._elem.modify(spec, width), self._length)

    def compiler_ctor(self):
        return 'v({}, {})'.format(self._elem.compiler_ctor(), self._length)

    def rust_name(self):
        return '{}x{}'.format(self._elem.rust_name(), self._length)

    def type_info(self, platform_info):
        elem_info = self._elem.type_info(platform_info)
        return elem_info.vectorize(self._length,
                                   platform_info.width_info(self.bitwidth()))

class Pointer(Type):
    def __init__(self, elem, llvm_elem, const):
        self._elem = elem;
        self._llvm_elem = llvm_elem
        self._const = const
        Type.__init__(self, BITWIDTH_POINTER)

    def modify(self, spec, width):
        if spec == 'D':
            return self._elem
        elif spec == 'M':
            return Pointer(self._elem, self._llvm_elem, False)
        elif spec == 'C':
            return Pointer(self._elem, self._llvm_elem, True)
        else:
            return Pointer(self._elem.modify(spec, width), self._llvm_elem, self._const)

    def compiler_ctor(self):
        if self._llvm_elem is None:
            llvm_elem = 'None'
        else:
            llvm_elem = 'Some({})'.format(self._llvm_elem.compiler_ctor())
        return 'p({}, {}, {})'.format('true' if self._const else 'false',
                                      self._elem.compiler_ctor(),
                                      llvm_elem)

    def rust_name(self):
        return '*{} {}'.format('const' if self._const else 'mut',
                               self._elem.rust_name())

    def type_info(self, platform_info):
        return self._elem.type_info(platform_info).pointer()

class Aggregate(Type):
    def __init__(self, flatten, elems):
        self._flatten = flatten
        self._elems = elems
        Type.__init__(self, sum(elem.bitwidth() for elem in elems))

    def __repr__(self):
        return '<Aggregate {}>'.format(self._elems)

    def compiler_ctor(self):
        return 'agg({}, vec![{}])'.format('true' if self._flatten else 'false',
                                          ', '.join(elem.compiler_ctor() for elem in self._elems))

    def rust_name(self):
        return '({})'.format(', '.join(elem.rust_name() for elem in self._elems))

    def type_info(self, platform_info):
        #return PlatformTypeInfo(None, None, self._llvm_name)
        return None


TYPE_ID_LOOKUP = {'i': [Signed, Unsigned],
                  's': [Signed],
                  'u': [Unsigned],
                  'f': [Float]}

def ptrify(match, elem, width, previous):
    ptr = match.group('pointer')
    if ptr is None:
        return elem
    else:
        llvm_ptr = match.group('llvm_pointer')
        if llvm_ptr is None:
            llvm_elem = None
        else:
            assert llvm_ptr.startswith('/')
            options = list(TypeSpec(llvm_ptr[1:]).enumerate(width, previous))
            assert len(options) == 1
            llvm_elem = options[0]
        assert ptr in ('Pc', 'Pm')
        return Pointer(elem, llvm_elem, ptr == 'Pc')

class TypeSpec(object):
    def __init__(self, spec):
        if not isinstance(spec, list):
            spec = [spec]

        self.spec = spec

    def enumerate(self, width, previous):
        for spec in self.spec:
            match = SPEC.match(spec)
            if match is not None:
                id = match.group('id')
                reference = match.group('reference')

                if id is not None:
                    is_vector = id.islower()
                    type_ctors = TYPE_ID_LOOKUP[id.lower()]

                    start = match.group('start')
                    if start is not None:
                        end = match.group('end')
                        llvm_width = None
                    else:
                        start = end = match.group('width')
                        llvm_width = match.group('llvm_width')
                    start = int(start)
                    end = int(end)

                    bitwidth = start
                    while bitwidth <= end:
                        for ctor in type_ctors:
                            if llvm_width is not None:
                                assert not is_vector
                                llvm_width = int(llvm_width)
                                assert llvm_width < bitwidth
                                scalar = ctor(bitwidth, llvm_width)
                            else:
                                scalar = ctor(bitwidth)

                            if is_vector:
                                elem = Vector(scalar, width // bitwidth)
                            else:
                                elem = scalar
                            yield ptrify(match, elem, width, previous)
                        bitwidth *= 2
                elif reference is not None:
                    reference = int(reference)
                    assert reference < len(previous), \
                        'referring to argument {}, but only {} are known'.format(reference,
                                                                                 len(previous))
                    ret = previous[reference]
                    for x in match.group('modifiers') or []:
                        ret = ret.modify(x, width)
                    force = match.group('force_width')
                    if force is not None:
                        ret = ret.modify(force, width)
                    yield ptrify(match, ret, width, previous)
                else:
                    assert False, 'matched `{}`, but didn\'t understand it?'.format(spec)
            elif spec.startswith('('):
                if spec.endswith(')'):
                    raise NotImplementedError()
                elif spec.endswith(')f'):
                    true_spec = spec[1:-2]
                    flatten = True

                for elems in itertools.product(*(TypeSpec(subspec).enumerate(width, previous)
                                                 for subspec in true_spec.split(','))):
                    yield Aggregate(flatten, elems)
            else:
                assert False, 'Failed to parse `{}`'.format(spec)

class GenericIntrinsic(object):
    def __init__(self, platform, intrinsic, widths, llvm_name, ret, args):
        self._platform = platform
        self.intrinsic = intrinsic
        self.widths = map(int, widths)
        self.llvm_name = llvm_name
        self.ret = TypeSpec(ret)
        self.args = list(map(TypeSpec, args))

    def monomorphise(self):
        for width in self.widths:
            # must be a power of two
            assert width & (width - 1) == 0
            def recur(processed, untouched):
                if untouched == []:
                    ret = processed[0]
                    args = processed[1:]
                    yield MonomorphicIntrinsic(self._platform, self.intrinsic, width,
                                               self.llvm_name,
                                               ret, args)
                else:
                    raw_arg = untouched[0]
                    rest = untouched[1:]
                    for arg in raw_arg.enumerate(width, processed):
                        for intr in recur(processed + [arg], rest):
                            yield intr

            for x in recur([], [self.ret] + self.args):
                yield x

class MonomorphicIntrinsic(object):
    def __init__(self, platform, intrinsic, width, llvm_name, ret, args):
        self._platform = platform
        self._intrinsic = intrinsic
        self._width = '' if width == 64 else 'q'
        self._llvm_name = llvm_name
        self._ret_raw = ret
        self._ret = ret.type_info(platform)
        self._args_raw = args
        self._args = [arg.type_info(platform) for arg in args]

    def llvm_name(self):
        if self._llvm_name.startswith('!'):
            return self._llvm_name[1:].format(self._ret, *self._args)
        else:
            return self._platform.llvm_prefix() + self._llvm_name.format(self._ret, *self._args)

    def intrinsic_suffix(self):
        return self._intrinsic.format(self._ret,
                                      *self._args,
                                      width = self._width)

    def intrinsic_name(self):
        return self._platform.platform().intrinsic_prefix() + self.intrinsic_suffix()

    def compiler_args(self):
        return ', '.join(arg.compiler_ctor() for arg in self._args_raw)

    def compiler_ret(self):
        return self._ret_raw.compiler_ctor()

    def compiler_signature(self):
        return '({}) -> {}'.format(self.compiler_args(), self.compiler_ret())

    def intrinsic_signature(self):
        names = 'xyzwabcdef'
        return '({}) -> {}'.format(', '.join('{}: {}'.format(name, arg.rust_name())
                                             for name, arg in zip(names, self._args_raw)),
                                   self._ret_raw.rust_name())

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = 'Render an intrinsic definition JSON to various formats.',
        epilog = textwrap.dedent('''\
        An intrinsic definition consists of a map with fields:
        - intrinsic: pattern for the name(s) of the vendor's C intrinsic(s)
        - llvm: pattern for the name(s) of the internal llvm intrinsic(s)
        - width: a vector of vector bit-widths the pattern works with
        - ret: type specifier for the return value
        - arguments: vector of type specifiers for arguments

        The width and types describe a range of possible intrinsics,
        and these are fed back into the intrinsic and llvm patterns to
        create the appropriate definitions.

        ## Type specifier grammar

        ```
        type := ( vector | scalar | aggregate | reference ) pointer?

        pointer := 'Pm' llvm_pointer? | 'Pc' llvm_pointer?
        llvm_pointer := '/' type

        vector := vector_elem width |
        vector_elem := 'i' | 'u' | 's' | 'f'

        scalar := scalar_type number llvm_width?
        scalar_type := 'U' | 'S' | 'F'
        llvm_width := '/' number

        aggregate := '(' (type),* ')' 'f'?

        reference := number modifiers*
        modifiers := 'v' | 'h' | 'd' | 'n' | 'w' | 'u' | 's' |
                     'x' number


        width = number | '(' number '-' number ')'

        number = [0-9]+
        ```

        ## Pointers

        Pointers can be created to any type. The `m` vs. `c` chooses
        mut vs. const. e.g. `S32Pm` corresponds to `*mut i32`, and
        `i32Pc` corresponds (with width 128) to `*const i8x16`,
        `*const u32x4`, etc.

        The type after the `/` (optional) represents the type used
        internally to LLVM, e.g. `S32pm/S8` is exposed as `*mut i32`
        in Rust, but is `i8*` in LLVM. (This defaults to the main
        type).

        ## Vectors

        The vector grammar is a pattern describing many possibilities
        for arguments/return value. The `vector_elem` describes the
        types of elements to use, and the `width` describes the (range
        of) widths for those elements, which are then placed into a
        vector with the `width` bitwidth. E.g. if an intrinsic has a
        `width` that includes 128, and the return value is `i(8-32)`,
        then some instantiation of that intrinsic will be `u8x16`,
        `u32x4`, `i32x4`, etc.

        ### Elements

        - i: integer, both signed and unsigned
        - u: unsigned integer
        - s: signed integer
        - f: float

        ## Scalars

        Similar to vectors, but these describe a single concrete type,
        not a range. The number is the bitwidth. The optional
        `llvm_width` is the bitwidth of the integer that should be
        passed to LLVM (by truncating the Rust argument): this only
        works with scalar integers and the LLVM width must be smaller
        than the Rust width.

        ### Types

        - U: unsigned integer
        - S: signed integer
        - F: float

        ## Aggregates

        An aggregate is a collection of multiple types; a tuple in
        Rust terms, or an unnamed struct in LLVM. The `f` modifiers
        forces the tuple to be flattened in the LLVM
        intrinsic. E.g. if `llvm.foo` takes `(F32,S32)`:

        - no `f` corresponds to `declare ... @llvm.foo({float, i32})`.
        - having an `f` corresponds to `declare ... @llvm.foo(float, i32)`.


        (Currently aggregates can not contain other aggregates.)

        ## References

        A reference uses the type of another argument, with possible
        modifications. The number refers to the type to use, starting
        with 0 == return value, 1 == first argument, 2 == second
        argument, etc.

        ### Modifiers

        - 'v': put a scalar into a vector of the current width (u32 -> u32x4, when width == 128)
        - 'h': half the length of the vector (u32x4 -> u32x2)
        - 'd': double the length of the vector (u32x2 -> u32x4)
        - 'n': narrow the element of the vector (u32x4 -> u16x4)
        - 'w': widen the element of the vector (u16x4 -> u32x4)
        - 'u': force an integer (vector or scalar) to be unsigned (i32x4 -> u32x4)
        - 's': force an integer (vector or scalar) to be signed (u32x4 -> i32x4)
        - 'x' number: force the type to be a vector of bitwidth `number`.
        - 'D': dereference a pointer (*mut u32 -> u32)
        - 'C': make a pointer const (*mut u32 -> *const u32)
        - 'M': make a pointer mut (*const u32 -> *mut u32)
        '''))
    parser.add_argument('--format', choices=FORMATS, required=True,
                        help = 'Output format.')
    parser.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                        help = 'File to output to (default stdout).')
    parser.add_argument('-i', '--info', type=argparse.FileType('r'),
                        help = 'File containing platform specific information to merge into'
                                'the input files\' header.')
    parser.add_argument('in_', metavar="FILE", type=argparse.FileType('r'), nargs='+',
                        help = 'JSON files to load')
    return parser.parse_args()


class ExternBlock(object):
    def __init__(self):
        pass

    def open(self, platform):
        return 'extern "platform-intrinsic" {'

    def render(self, mono):
        return '    fn {}{};'.format(mono.intrinsic_name(),
                                     mono.intrinsic_signature())

    def close(self):
        return '}'

class CompilerDefs(object):
    def __init__(self):
        pass

    def open(self, platform):
        return '''\
// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// DO NOT EDIT: autogenerated by etc/platform-intrinsics/generator.py
// ignore-tidy-linelength

#![allow(unused_imports)]

use {{Intrinsic, i, i_, u, u_, f, v, agg, p}};
use IntrinsicDef::Named;
use rustc::middle::ty;

pub fn find<'tcx>(_tcx: &ty::ctxt<'tcx>, name: &str) -> Option<Intrinsic> {{
    if !name.starts_with("{0}") {{ return None }}
    Some(match &name["{0}".len()..] {{'''.format(platform.intrinsic_prefix())

    def render(self, mono):
        return '''\
        "{}" => Intrinsic {{
            inputs: vec![{}],
            output: {},
            definition: Named("{}")
        }},'''.format(mono.intrinsic_suffix(),
                      mono.compiler_args(),
                      mono.compiler_ret(),
                      mono.llvm_name())

    def close(self):
        return '''\
        _ => return None,
    })
}'''

FORMATS = {
    'extern-block': ExternBlock(),
    'compiler-defs': CompilerDefs(),
}


def main():
    args = parse_args()
    ins = args.in_
    out = args.out
    out_format = FORMATS[args.format]
    info = args.info
    one_file_no_info = False
    if len(ins) > 1 and info is None:
        print('error: cannot have multiple inputs without an info header.', file=sys.stderr)
        sys.exit(1)

    elif info is None:
        info = ins[0]
        one_file_no_info = True
    info_json = json.load(info)
    platform = PlatformInfo(info_json)

    print(out_format.open(platform), file=out)

    for in_ in ins:

        if one_file_no_info:
            data = info_json
        else:
            data = json.load(in_)
            data.update(info_json)

        intrinsics = IntrinsicSet(platform, data)
        for intr in intrinsics.intrinsics():
            for mono in intr.monomorphise():
                print(out_format.render(mono), file=out)

    print(out_format.close(), file=out)

if __name__ == '__main__':
    main()
