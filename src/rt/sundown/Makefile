# Makefile

# Copyright (c) 2009, Natacha Porté
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

DEPDIR=depends

# "Machine-dependent" options
#MFLAGS=-fPIC

CFLAGS=-c -g -O3 -fPIC -Wall -Werror -Wsign-compare -Isrc -Ihtml
LDFLAGS=-g -O3 -Wall -Werror 
CC=gcc


SUNDOWN_SRC=\
	src/markdown.o \
	src/stack.o \
	src/buffer.o \
	src/autolink.o \
	html/html.o \
	html/html_smartypants.o \
	html/houdini_html_e.o \
	html/houdini_href_e.o

all:		libsundown.so sundown smartypants html_blocks

.PHONY:		all clean

# libraries

libsundown.so:	libsundown.so.1
	ln -f -s $^ $@

libsundown.so.1: $(SUNDOWN_SRC)
	$(CC) $(LDFLAGS) -shared -Wl $^ -o $@

# executables

sundown:	examples/sundown.o $(SUNDOWN_SRC)
	$(CC) $(LDFLAGS) $^ -o $@

smartypants: examples/smartypants.o $(SUNDOWN_SRC)
	$(CC) $(LDFLAGS) $^ -o $@

# perfect hashing
html_blocks: src/html_blocks.h

src/html_blocks.h: html_block_names.txt
	gperf -N find_block_tag -H hash_block_tag -C -c -E --ignore-case $^ > $@


# housekeeping
clean:
	rm -f src/*.o html/*.o examples/*.o
	rm -f libsundown.so libsundown.so.1 sundown smartypants
	rm -f sundown.exe smartypants.exe
	rm -rf $(DEPDIR)


# dependencies

include $(wildcard $(DEPDIR)/*.d)


# generic object compilations

%.o:	src/%.c examples/%.c html/%.c
	@mkdir -p $(DEPDIR)
	@$(CC) -MM $< > $(DEPDIR)/$*.d
	$(CC) $(CFLAGS) -o $@ $<

