# Copyright Joyent, Inc. and other Node contributors. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# Use make -f Makefile.gcc PREFIX=i686-w64-mingw32-
# for cross compilation
CC ?= $(PREFIX)gcc
AR ?= $(PREFIX)ar
E=.exe

CFLAGS+=$(CPPFLAGS) -g --std=gnu89 -D_WIN32_WINNT=0x0501 -I$(S)src/ares/config_win32
LINKFLAGS=-lm

CARES_OBJS += src/ares/windows_port.o

uv.a: src/uv-win.o src/uv-common.o src/uv-eio.o src/eio/eio.o $(CARES_OBJS)
	@$(call EE, ar: $@)
	$(Q)$(AR) rcs uv.a $^

src/uv-win.o: src/uv-win.c include/uv.h include/uv-win.h
	@$(call EE, compile: $@)
	$(Q)$(CC) $(CFLAGS) -c $< -o $@

src/uv-common.o: src/uv-common.c include/uv.h include/uv-win.h
	@$(call EE, compile: $@)
	$(Q)$(CC) $(CFLAGS) -c $< -o $@

EIO_CPPFLAGS += $(CPPFLAGS)
EIO_CPPFLAGS += -DEIO_CONFIG_H=\"$(EIO_CONFIG)\"
EIO_CPPFLAGS += -DEIO_STACKSIZE=65536
EIO_CPPFLAGS += -D_GNU_SOURCE

src/eio/eio.o: src/eio/eio.c
	@$(call EE, compile: $@)
	$(Q)$(CC) $(EIO_CPPFLAGS) $(CFLAGS) -c $< -o $@

src/uv-eio.o: src/uv-eio.c
	@$(call EE, compile: $@)
	$(Q)$(CC) $(CPPFLAGS) -I$(S)src/eio/ $(CFLAGS) -c $< -o $@

