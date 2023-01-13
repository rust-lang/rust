include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) rs.rs -Z unstable-options -l static=l1 -l link-arg=a1 -l static=l2 -l link-arg=a2 -l dylib=d1 -l link-arg=a3 --print link-args | $(CGREP) -e 'l1.*a1.*l2.*a2.*d1.*a3'
