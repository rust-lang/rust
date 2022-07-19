-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) rs.rs -l static=l1 -l link-arg=a1 -l static=l2 -l link-arg=a2 -l dylib=d1 -l link-arg=a3 --print link-args | $(CGREP) -e '-ll1.*a1.*-ll2.*a2.*-ld1.*a3'
