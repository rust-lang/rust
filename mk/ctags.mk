# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

######################################################################
# TAGS file creation.  No dependency tracking, just do it on demand.
# Requires Exuberant Ctags: http://ctags.sourceforge.net/index.html
######################################################################

.PHONY: TAGS.emacs TAGS.vi

# This is using a blacklist approach, probably more durable than a whitelist.
# We exclude: external dependencies (llvm, rt/{msvc,sundown,vg}),
# tests (compiletest, test) and a couple of other things (rt/arch, etc)
CTAGS_LOCATIONS=$(patsubst ${CFG_SRC_DIR}src/llvm,, \
				$(patsubst ${CFG_SRC_DIR}src/compiletest,, \
				$(patsubst ${CFG_SRC_DIR}src/test,, \
				$(patsubst ${CFG_SRC_DIR}src/etc,, \
				$(patsubst ${CFG_SRC_DIR}src/rt,, \
				$(patsubst ${CFG_SRC_DIR}src/rt/arch,, \
				$(patsubst ${CFG_SRC_DIR}src/rt/msvc,, \
				$(patsubst ${CFG_SRC_DIR}src/rt/sundown,, \
				$(patsubst ${CFG_SRC_DIR}src/rt/vg,, \
				$(wildcard ${CFG_SRC_DIR}src/*) $(wildcard ${CFG_SRC_DIR}src/rt/*) \
				)))))))))
CTAGS_OPTS=--options="${CFG_SRC_DIR}src/etc/ctags.rust" --languages=-javascript --recurse ${CTAGS_LOCATIONS}
# We could use `--languages=Rust`, but there is value in producing tags for the
# C++ parts of the code base too (at the time of writing, those are .h and .cpp
# files in src/rt, src/rt/sync and src/rustllvm); we mainly just want to
# exclude the external dependencies.

TAGS.emacs:
	ctags -e -f $@ ${CTAGS_OPTS}

TAGS.vi:
	ctags -f $@ ${CTAGS_OPTS}
