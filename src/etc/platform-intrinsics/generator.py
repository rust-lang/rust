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
    r'^(?:(?P<void>V)|(?P<id>[iusfIUSF])(?:\((?P<start>\d+)-(?P<end>\d+)\)|'
    r'(?P<width>\d+)(:?/(?P<llvm_width>\d+))?)'
    r'|(?P<reference>\d+))(?P<index>\.\d+)?(?P<modifiers>[vShdnwusfDMC]*)(?P<force_width>x\d+)?'
    r'(?:(?P<pointer>Pm|Pc)(?P<llvm_pointer>/.*)?|(?P<bitcast>->.*))?$'
)

class PlatformInfo(object):
    def __init__(self, json):
        self._platform = json['platform']

    def platform_prefix(self):
        return self._platform

class IntrinsicSet(object):
    def __init__(self, platform, json):
        self._llvm_prefix = json['llvm_prefix']
        self._type_info = json['number_info']
        self._intrinsics = json['intrinsics']
        self._widths = json['width_info']
        self._platform = platform
        self._intrinsic_prefix = json['intrinsic_prefix']

    def intrinsics(self):
        for raw in self._intrinsics:
            yield GenericIntrinsic(self,
                                   raw['intrinsic'], raw['width'], raw['llvm'],
                                   raw['ret'], raw['args'])

    def platform(self):
        return self._platform

    def intrinsic_prefix(self):
        return self._intrinsic_prefix

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
    def __init__(self, llvm_name, properties, elems = None):
        if elems is None:
            self.properties = properties
            self.llvm_name = llvm_name
        else:
            assert properties is None and llvm_name is None
            self.properties = {}
            self.elems = elems

    def __repr__(self):
        return '<PlatformTypeInfo {}, {}>'.format(self.llvm_name, self.properties)

    def __getattr__(self, name):
        return self.properties[name]

    def __getitem__(self, idx):
        return self.elems[idx]

    def vectorize(self, length, width_info):
        props = self.properties.copy()
        props.update(width_info)
        return PlatformTypeInfo('v{}{}'.format(length, self.llvm_name), props)

    def pointer(self, llvm_elem):
        name = self.llvm_name if llvm_elem is None else llvm_elem.llvm_name
        return PlatformTypeInfo('p0{}'.format(name), self.properties)

BITWIDTH_POINTER = '<pointer>'

class Type(object):
    def __init__(self, bitwidth):
        self._bitwidth = bitwidth

    def bitwidth(self):
        return self._bitwidth

    def modify(self, spec, width, previous):
        raise NotImplementedError()

    def __ne__(self, other):
        return not (self == other)

class Void(Type):
    def __init__(self):
        Type.__init__(self, 0)

    @staticmethod
    def compiler_ctor():
        return '::VOID'

    def compiler_ctor_ref(self):
        return '&' + self.compiler_ctor()

    @staticmethod
    def rust_name():
        return '()'

    @staticmethod
    def type_info(platform_info):
        return None

    def __eq__(self, other):
        return isinstance(other, Void)

class Number(Type):
    def __init__(self, bitwidth):
        Type.__init__(self, bitwidth)

    def modify(self, spec, width, previous):
        if spec == 'u':
            return Unsigned(self.bitwidth())
        elif spec == 's':
            return Signed(self.bitwidth())
        elif spec == 'f':
            return Float(self.bitwidth())
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

    def __eq__(self, other):
        # print(self, other)
        return self.__class__ == other.__class__ and self.bitwidth() == other.bitwidth()

class Signed(Number):
    def __init__(self, bitwidth, llvm_bitwidth = None):
        Number.__init__(self, bitwidth)
        self._llvm_bitwidth = llvm_bitwidth


    def compiler_ctor(self):
        if self._llvm_bitwidth is None:
            return '::I{}'.format(self.bitwidth())
        else:
            return '::I{}_{}'.format(self.bitwidth(), self._llvm_bitwidth)

    def compiler_ctor_ref(self):
        return '&' + self.compiler_ctor()

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
            return '::U{}'.format(self.bitwidth())
        else:
            return '::U{}_{}'.format(self.bitwidth(), self._llvm_bitwidth)

    def compiler_ctor_ref(self):
        return '&' + self.compiler_ctor()

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
        return '::F{}'.format(self.bitwidth())

    def compiler_ctor_ref(self):
        return '&' + self.compiler_ctor()

    def llvm_name(self):
        return 'f{}'.format(self.bitwidth())

    def rust_name(self):
        return 'f{}'.format(self.bitwidth())

class Vector(Type):
    def __init__(self, elem, length, bitcast = None):
        assert isinstance(elem, Type) and not isinstance(elem, Vector)
        Type.__init__(self,
                      elem.bitwidth() * length)
        self._length = length
        self._elem = elem
        assert bitcast is None or (isinstance(bitcast, Vector) and
                                   bitcast._bitcast is None and
                                   bitcast._elem.bitwidth() == elem.bitwidth())
        if bitcast is not None and bitcast._elem != elem:
            self._bitcast = bitcast._elem
        else:
            self._bitcast = None

    def modify(self, spec, width, previous):
        if spec == 'S':
            return self._elem
        elif spec == 'h':
            return Vector(self._elem, self._length // 2)
        elif spec == 'd':
            return Vector(self._elem, self._length * 2)
        elif spec.startswith('x'):
            new_bitwidth = int(spec[1:])
            return Vector(self._elem, new_bitwidth // self._elem.bitwidth())
        elif spec.startswith('->'):
            bitcast_to = TypeSpec(spec[2:])
            choices = list(bitcast_to.enumerate(width, previous))
            assert len(choices) == 1
            bitcast_to = choices[0]
            return Vector(self._elem, self._length, bitcast_to)
        else:
            return Vector(self._elem.modify(spec, width, previous), self._length)

    def compiler_ctor(self):
        if self._bitcast is None:
            return '{}x{}'.format(self._elem.compiler_ctor(),
                                     self._length)
        else:
            return '{}x{}_{}'.format(self._elem.compiler_ctor(),
                                     self._length,
                                     self._bitcast.compiler_ctor()
                                         .replace('::', ''))

    def compiler_ctor_ref(self):
        return '&' + self.compiler_ctor()

    def rust_name(self):
        return '{}x{}'.format(self._elem.rust_name(), self._length)

    def type_info(self, platform_info):
        elem_info = self._elem.type_info(platform_info)
        return elem_info.vectorize(self._length,
                                   platform_info.width_info(self.bitwidth()))

    def __eq__(self, other):
        return isinstance(other, Vector) and self._length == other._length and \
            self._elem == other._elem and self._bitcast == other._bitcast

class Pointer(Type):
    def __init__(self, elem, llvm_elem, const):
        self._elem = elem
        self._llvm_elem = llvm_elem
        self._const = const
        Type.__init__(self, BITWIDTH_POINTER)

    def modify(self, spec, width, previous):
        if spec == 'D':
            return self._elem
        elif spec == 'M':
            return Pointer(self._elem, self._llvm_elem, False)
        elif spec == 'C':
            return Pointer(self._elem, self._llvm_elem, True)
        else:
            return Pointer(self._elem.modify(spec, width, previous), self._llvm_elem, self._const)

    def compiler_ctor(self):
        if self._llvm_elem is None:
            llvm_elem = 'None'
        else:
            llvm_elem = 'Some({})'.format(self._llvm_elem.compiler_ctor_ref())
        return 'Type::Pointer({}, {}, {})'.format(self._elem.compiler_ctor_ref(),
                                                  llvm_elem,
                                                  'true' if self._const else 'false')

    def compiler_ctor_ref(self):
        return "{{ static PTR: Type = {}; &PTR }}".format(self.compiler_ctor())


    def rust_name(self):
        return '*{} {}'.format('const' if self._const else 'mut',
                               self._elem.rust_name())

    def type_info(self, platform_info):
        if self._llvm_elem is None:
            llvm_elem = None
        else:
            llvm_elem = self._llvm_elem.type_info(platform_info)
        return self._elem.type_info(platform_info).pointer(llvm_elem)

    def __eq__(self, other):
        return isinstance(other, Pointer) and self._const == other._const \
            and self._elem == other._elem and self._llvm_elem == other._llvm_elem

class Aggregate(Type):
    def __init__(self, flatten, elems):
        self._flatten = flatten
        self._elems = elems
        Type.__init__(self, sum(elem.bitwidth() for elem in elems))

    def __repr__(self):
        return '<Aggregate {}>'.format(self._elems)

    def modify(self, spec, width, previous):
        if spec.startswith('.'):
            num = int(spec[1:])
            return self._elems[num]
        else:
            print(spec)
            raise NotImplementedError()

    def compiler_ctor(self):
        parts = "{{ static PARTS: [&'static Type; {}] = [{}]; &PARTS }}"
        elems = ', '.join(elem.compiler_ctor_ref() for elem in self._elems)
        parts = parts.format(len(self._elems), elems)
        return 'Type::Aggregate({}, {})'.format('true' if self._flatten else 'false',
                                                parts)

    def compiler_ctor_ref(self):
        return "{{ static AGG: Type = {}; &AGG }}".format(self.compiler_ctor())

    def rust_name(self):
        return '({})'.format(', '.join(elem.rust_name() for elem in self._elems))

    def type_info(self, platform_info):
        return PlatformTypeInfo(None, None, [elem.type_info(platform_info) for elem in self._elems])

    def __eq__(self, other):
        return isinstance(other, Aggregate) and self._flatten == other._flatten and \
            self._elems == other._elems


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

                modifiers = []
                index = match.group('index')
                if index is not None:
                    modifiers.append(index)
                modifiers += list(match.group('modifiers') or '')
                force = match.group('force_width')
                if force is not None:
                    modifiers.append(force)
                bitcast = match.group('bitcast')
                if bitcast is not None:
                    modifiers.append(bitcast)

                if match.group('void') is not None:
                    assert spec == 'V'
                    yield Void()
                elif id is not None:
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
                                assert bitcast is None
                                elem = scalar

                            for x in modifiers:
                                elem = elem.modify(x, width, previous)
                            yield ptrify(match, elem, width, previous)
                        bitwidth *= 2
                elif reference is not None:
                    reference = int(reference)
                    assert reference < len(previous), \
                        'referring to argument {}, but only {} are known'.format(reference,
                                                                                 len(previous))
                    ret = previous[reference]
                    for x in modifiers:
                        ret = ret.modify(x, width, previous)
                    yield ptrify(match, ret, width, previous)
                else:
                    assert False, 'matched `{}`, but didn\'t understand it?'.format(spec)
            elif spec.startswith('('):
                if spec.endswith(')'):
                    true_spec = spec[1:-1]
                    flatten = False
                elif spec.endswith(')f'):
                    true_spec = spec[1:-2]
                    flatten = True
                else:
                    assert False, 'found unclosed aggregate `{}`'.format(spec)

                for elems in itertools.product(*(TypeSpec(subspec).enumerate(width, previous)
                                                 for subspec in true_spec.split(','))):
                    yield Aggregate(flatten, elems)
            elif spec.startswith('['):
                if spec.endswith(']'):
                    true_spec = spec[1:-1]
                    flatten = False
                elif spec.endswith(']f'):
                    true_spec = spec[1:-2]
                    flatten = True
                else:
                    assert False, 'found unclosed aggregate `{}`'.format(spec)
                elem_spec, count = true_spec.split(';')

                count = int(count)
                for elem in TypeSpec(elem_spec).enumerate(width, previous):
                    yield Aggregate(flatten, [elem] * count)
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
                if not untouched:
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

    def platform_prefix(self):
        return self._platform.platform().platform_prefix()

    def intrinsic_set_name(self):
        return self._platform.intrinsic_prefix()

    def intrinsic_name(self):
        return self._platform.intrinsic_prefix() + self.intrinsic_suffix()

    def compiler_args(self):
        return ', '.join(arg.compiler_ctor_ref() for arg in self._args_raw)

    def compiler_ret(self):
        return self._ret_raw.compiler_ctor_ref()

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
        Quick How-To:

        There are two operating modes: single file and multiple files.

        For example, ARM is specified as a single file. To generate the
        compiler-definitions for ARM just pass the script the "arm.json" file:

        python generator.py --format compiler-defs arm.json

        The X86 architecture is specified as multiple files (for the different
        instruction sets that x86 supports). To generate the compiler
        definitions one needs to pass the script a "platform information file"
        (with the -i flag) next to the files of the different intruction sets.
        For example, to generate the X86 compiler-definitions for SSE4.2, just:

        python generator.py --format compiler-defs -i x86/info.json sse42.json

        And to generate the compiler-definitions for SSE4.1 and SSE4.2, just:

        python generator.py --format compiler-defs -i x86/info.json sse41.json sse42.json

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
        type := core_type modifier* suffix?

        core_type := void | vector | scalar | aggregate | reference

        modifier := 'v' | 'h' | 'd' | 'n' | 'w' | 'u' | 's' |
                     'x' number | '.' number
        suffix := pointer | bitcast
        pointer := 'Pm' llvm_pointer? | 'Pc' llvm_pointer?
        llvm_pointer := '/' type
        bitcast := '->' type

        void := 'V'

        vector := vector_elem width |
        vector_elem := 'i' | 'u' | 's' | 'f'

        scalar := scalar_type number llvm_width?
        scalar_type := 'U' | 'S' | 'F'
        llvm_width := '/' number

        aggregate := '(' (type),* ')' 'f'? | '[' type ';' number ']' 'f'?

        reference := number

        width = number | '(' number '-' number ')'

        number = [0-9]+
        ```

        ## Void

        The `V` type corresponds to `void` in LLVM (`()` in
        Rust). It's likely to only work in return position.

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

        The `[type;number]` form is a just shorter way to write
        `(...)`, except avoids doing a cartesian product of generic
        types, e.g. `[S32;2]` is the same as `(S32, S32)`, while
        `[I32;2]` is describing just the two types `(S32,S32)` and
        `(U32,U32)` (i.e. doesn't include `(S32,U32)`, `(U32,S32)` as
        `(I32,I32)` would).

        (Currently aggregates can not contain other aggregates.)

        ## References

        A reference uses the type of another argument, with possible
        modifications. The number refers to the type to use, starting
        with 0 == return value, 1 == first argument, 2 == second
        argument, etc.

        ## Affixes

        The `modifier` and `suffix` adaptors change the precise
        representation.

        ### Modifiers

        - 'v': put a scalar into a vector of the current width (u32 -> u32x4, when width == 128)
        - 'S': get the scalar element of a vector (u32x4 -> u32)
        - 'h': half the length of the vector (u32x4 -> u32x2)
        - 'd': double the length of the vector (u32x2 -> u32x4)
        - 'n': narrow the element of the vector (u32x4 -> u16x4)
        - 'w': widen the element of the vector (u16x4 -> u32x4)
        - 'u': force a number (vector or scalar) to be unsigned int (f32x4 -> u32x4)
        - 's': force a number (vector or scalar) to be signed int (u32x4 -> i32x4)
        - 'f': force a number (vector or scalar) to be float (u32x4 -> f32x4)
        - 'x' number: force the type to be a vector of bitwidth `number`.
        - '.' number: get the `number`th element of an aggregate
        - 'D': dereference a pointer (*mut u32 -> u32)
        - 'C': make a pointer const (*mut u32 -> *const u32)
        - 'M': make a pointer mut (*const u32 -> *mut u32)

        ### Pointers

        Pointers can be created of any type by appending a `P*`
        suffix. The `m` vs. `c` chooses mut vs. const. e.g. `S32Pm`
        corresponds to `*mut i32`, and `i32Pc` corresponds (with width
        128) to `*const i8x16`, `*const u32x4`, etc.

        The type after the `/` (optional) represents the type used
        internally to LLVM, e.g. `S32pm/S8` is exposed as `*mut i32`
        in Rust, but is `i8*` in LLVM. (This defaults to the main
        type).

        ### Bitcast

        The `'->' type` bitcast suffix will cause the value to be
        bitcast to the right-hand type when calling the intrinsic,
        e.g. `s32->f32` will expose the intrinsic as `i32x4` at the
        Rust level, but will cast that vector to `f32x4` when calling
        the LLVM intrinsic.
        '''))
    parser.add_argument('--format', choices=FORMATS, required=True,
                        help = 'Output format.')
    parser.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                        help = 'File to output to (default stdout).')
    parser.add_argument('-i', '--info', type=argparse.FileType('r'),
                        help = 'File containing platform specific information to merge into '
                                'the input files\' header.')
    parser.add_argument('in_', metavar="FILE", type=argparse.FileType('r'), nargs='+',
                        help = 'JSON files to load')
    return parser.parse_args()


class ExternBlock(object):
    def __init__(self):
        pass

    @staticmethod
    def open(platform):
        return 'extern "platform-intrinsic" {'

    @staticmethod
    def render(mono):
        return '    fn {}{}{};'.format(mono.platform_prefix(),
                                       mono.intrinsic_name(),
                                       mono.intrinsic_signature())

    @staticmethod
    def close():
        return '}'

class CompilerDefs(object):
    def __init__(self):
        pass

    @staticmethod
    def open(platform):
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

use {{Intrinsic, Type}};
use IntrinsicDef::Named;

// The default inlining settings trigger a pathological behaviour in
// LLVM, which causes makes compilation very slow. See #28273.
#[inline(never)]
pub fn find(name: &str) -> Option<Intrinsic> {{
    if !name.starts_with("{0}") {{ return None }}
    Some(match &name["{0}".len()..] {{'''.format(platform.platform_prefix())

    @staticmethod
    def render(mono):
        return '''\
        "{}" => Intrinsic {{
            inputs: {{ static INPUTS: [&'static Type; {}] = [{}]; &INPUTS }},
            output: {},
            definition: Named("{}")
        }},'''.format(mono.intrinsic_set_name() + mono.intrinsic_suffix(),
                      len(mono._args_raw),
                      mono.compiler_args(),
                      mono.compiler_ret(),
                      mono.llvm_name())

    @staticmethod
    def close():
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
