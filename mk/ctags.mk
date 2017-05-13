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

CTAGS_RUSTC_LOCATIONS=$(patsubst ${CFG_SRC_DIR}src/lib%test,, \
				$(wildcard ${CFG_SRC_DIR}src/lib*)) ${CFG_SRC_DIR}src/libtest
CTAGS_LOCATIONS=$(patsubst ${CFG_SRC_DIR}src/librust%,, \
                $(patsubst ${CFG_SRC_DIR}src/lib%test,, \
				$(wildcard ${CFG_SRC_DIR}src/lib*))) ${CFG_SRC_DIR}src/libtest
CTAGS_OPTS=--options="${CFG_SRC_DIR}src/etc/ctags.rust" --languages=Rust --recurse

TAGS.rustc.emacs:
	ctags -e -f $@ ${CTAGS_OPTS} ${CTAGS_RUSTC_LOCATIONS}

TAGS.emacs:
	ctags -e -f $@ ${CTAGS_OPTS} ${CTAGS_LOCATIONS}

TAGS.rustc.vi:
	ctags -f $@ ${CTAGS_OPTS} ${CTAGS_RUSTC_LOCATIONS}

TAGS.vi:
	ctags -f $@ ${CTAGS_OPTS} ${CTAGS_LOCATIONS}
