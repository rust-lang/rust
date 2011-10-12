######################################################################
# TAGS file creation.  No dependency tracking, just do it on demand.
# Requires Exuberant Ctags: http://ctags.sourceforge.net/index.html
######################################################################

CTAGS_OPTS=--options=${CFG_SRC_DIR}/src/etc/ctags.rust -R ${CFG_SRC_DIR}/src

TAGS.emacs:
	ctags -e -f $@ ${CTAGS_OPTS}

TAGS.vi:
	ctags -f $@ ${CTAGS_OPTS}
