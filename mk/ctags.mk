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

CTAGS_OPTS=--options=${CFG_SRC_DIR}/src/etc/ctags.rust -R ${CFG_SRC_DIR}/src

TAGS.emacs:
	ctags -e -f $@ ${CTAGS_OPTS}

TAGS.vi:
	ctags -f $@ ${CTAGS_OPTS}
