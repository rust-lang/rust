# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

ifdef VERBOSE
  Q :=
  E =
else
  Q := @
  E = echo $(1)
endif

print-%:
	@echo $*=$($*)

S := $(CFG_SRC_DIR)
SREL := $(CFG_SRC_DIR_RELATIVE)

ifeq ($(CFG_OSTYPE),pc-windows-gnu)
  NACL_TOOLCHAIN_OS_PATH:=win
else ifeq ($(CFG_OSTYPE),apple-darwin)
  NACL_TOOLCHAIN_OS_PATH:=mac
else
  NACL_TOOLCHAIN_OS_PATH:=linux
endif

ifneq ("$(wildcard $(CFG_NACL_CROSS_PATH)/REV)","")
# The user is using a toolchain built from source, or otherwise pointed
# CFG_NACL_CROSS_PATH directly to the PNaCl toolchain root.
CFG_PNACL_TOOLCHAIN:=$(CFG_NACL_CROSS_PATH)
else
CFG_PNACL_TOOLCHAIN:=$(CFG_NACL_CROSS_PATH)/toolchain/$(NACL_TOOLCHAIN_OS_PATH)_pnacl
endif
