# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Recursive wildcard function
# http://blog.jgc.org/2011/07/gnu-make-recursive-wildcard-function.html
rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) \
  $(filter $(subst *,%,$2),$d))

ifndef CFG_DISABLE_MANAGE_SUBMODULES
# This is a pretty expensive operation but I don't see any way to avoid it
NEED_GIT_RECONFIG=$(shell cd "$(CFG_SRC_DIR)" && "$(CFG_GIT)" submodule status | grep -c '^\(+\|-\)')
else
NEED_GIT_RECONFIG=0
endif

ifeq ($(NEED_GIT_RECONFIG),0)
else
# If the submodules have changed then always execute config.mk
.PHONY: config.stamp
endif

Makefile config.mk: config.stamp

config.stamp: $(S)configure $(S)Makefile.in $(S)src/snapshots.txt
	@$(call E, cfg: reconfiguring)
	$(S)configure $(CFG_CONFIGURE_ARGS)
