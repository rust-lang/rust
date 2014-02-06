# Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

"""
Script for extracting compilable fragments from markdown documentation. See
prep.js for a description of the format recognized by this tool. Expects
a directory fragments/ to exist under the current directory, and writes the
fragments in there as individual .rs files.
"""
from __future__ import print_function
from codecs import open
from collections import deque
from itertools import imap
import os
import re
import sys

# regexes
CHAPTER_NAME_REGEX = re.compile(r'# (.*)')
CODE_BLOCK_DELIM_REGEX = re.compile(r'~~~')
COMMENT_REGEX = re.compile(r'^# ')
COMPILER_DIRECTIVE_REGEX = re.compile(r'\#\[(.*)\];')
ELLIPSES_REGEX = re.compile(r'\.\.\.')
EXTERN_MOD_REGEX = re.compile(r'\bextern mod extra\b')
MAIN_FUNCTION_REGEX = re.compile(r'\bfn main\b')
TAGS_REGEX = re.compile(r'\.([\w-]*)')

# tags to ignore
IGNORE_TAGS = \
        frozenset(["abnf", "ebnf", "field", "keyword", "notrust", "precedence"])

# header for code snippet files
OUTPUT_BLOCK_HEADER = '\n'.join((
    "#[ deny(warnings) ];",
    "#[ allow(unused_variable) ];",
    "#[ allow(dead_assignment) ];",
    "#[ allow(unused_mut) ];",
    "#[ allow(attribute_usage) ];",
    "#[ allow(dead_code) ];",
    "#[ feature(macro_rules, globs, struct_variant, managed_boxes) ];\n",))


def add_extern_mod(block):
    if not has_extern_mod(block):
        # add `extern mod extra;` after compiler directives
        directives = []
        while len(block) and is_compiler_directive(block[0]):
            directives.append(block.popleft())

        block.appendleft("\nextern mod extra;\n\n")
        block.extendleft(reversed(directives))

    return block


def add_main_function(block):
    if not has_main_function(block):
        prepend_spaces = lambda x: '    ' + x
        block = deque(imap(prepend_spaces, block))
        block.appendleft("\nfn main() {\n")
        block.append("\n}\n")
    return block


def extract_code_fragments(dest_dir, lines):
    """
    Extracts all the code fragments from a file that do not have ignored tags
    writing them to the following file:

        [dest dir]/[chapter name]_[chapter_index].rs
    """
    chapter_name = None
    chapter_index = 0

    for line in lines:
        if is_chapter_title(line):
            chapter_name = get_chapter_name(line)
            chapter_index = 1
            continue

        if not is_code_block_delim(line):
            continue

        assert chapter_name, "Chapter name missing for code block."
        tags = get_tags(line)
        block = get_code_block(lines)

        if tags & IGNORE_TAGS:
            continue

        block = add_extern_mod(add_main_function(block))
        block.appendleft(OUTPUT_BLOCK_HEADER)

        if "ignore" in tags:
            block.appendleft("//xfail-test\n")
        elif "should_fail" in tags:
            block.appendleft("//should-fail\n")

        output_filename = os.path.join(
                dest_dir,
                chapter_name + '_' + str(chapter_index) + '.rs')

        write_file(output_filename, block)
        chapter_index += 1


def has_extern_mod(block):
    """Checks if a code block has the line `extern mod extra`."""
    find_extern_mod = lambda x: re.search(EXTERN_MOD_REGEX, x)
    return any(imap(find_extern_mod, block))


def has_main_function(block):
    """Checks if a code block has a main function."""
    find_main_fn = lambda x: re.search(MAIN_FUNCTION_REGEX, x)
    return any(imap(find_main_fn, block))


def is_chapter_title(line):
    return re.match(CHAPTER_NAME_REGEX, line)


def is_code_block_delim(line):
    return re.match(CODE_BLOCK_DELIM_REGEX, line)


def is_compiler_directive(line):
    return re.match(COMPILER_DIRECTIVE_REGEX, line)


def get_chapter_name(line):
    """Get the chapter name from a `# Containers` line."""
    return re.sub(
            r'\W',
            '_',
            re.match(CHAPTER_NAME_REGEX, line).group(1)).lower()


def get_code_block(lines):
    """
    Get a code block surrounded by ~~~, for example:

        1: ~~~ { .tag }
        2: let u: ~[u32] = ~[0, 1, 2];
        3: let v: &[u32] = &[0, 1, 2, 3];
        4: let w: [u32, .. 5] = [0, 1, 2, 3, 4];
        5:
        6: println!("u: {}, v: {}, w: {}", u.len(), v.len(), w.len());
        7: ~~~

    Returns lines 2-6. Assumes line 1 has been consumed by the caller.
    """
    strip_comments = lambda x: re.sub(COMMENT_REGEX, '', x)
    strip_ellipses = lambda x: re.sub(ELLIPSES_REGEX, '', x)

    result = deque()

    for line in lines:
        if is_code_block_delim(line):
            break
        result.append(strip_comments(strip_ellipses(line)))
    return result


def get_lines(filename):
    with open(filename) as f:
        for line in f:
            yield line


def get_tags(line):
    """
    Retrieves all tags from the line format:
        ~~~ { .tag1 .tag2 .tag3 }
    """
    return set(re.findall(TAGS_REGEX, line))


def write_file(filename, lines):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(unicode(line, encoding='utf-8', errors='replace'))


def main(argv=None):
    if not argv:
        argv = sys.argv

    if len(sys.argv) < 2:
        sys.stderr.write("Please provide an input filename.")
        sys.exit(1)
    elif len(sys.argv) < 3:
        sys.stderr.write("Please provide a destination directory.")
        sys.exit(1)

    input_file = sys.argv[1]
    dest_dir = sys.argv[2]

    if not os.path.exists(input_file):
        sys.stderr.write("Input file does not exist.")
        sys.exit(1)

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    extract_code_fragments(dest_dir, get_lines(input_file))


if __name__ == "__main__":
    sys.exit(main())
