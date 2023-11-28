# only-linux

include ../tools.mk

all:
	$(RUSTC) rs.rs -Z unstable-options -C linker-flavor=gnu-cc -l static=l1 -l link-arg=a1 -l static=l2 -l link-arg=a2 -l dylib=d1 -l link-arg=a3 --print link-args | $(CGREP) -e 'l1.*-Wl,a1.*l2.*-Wl,a2.*d1.*-Wl,a3'
	$(RUSTC) rs.rs -Z unstable-options -C linker-flavor=gnu-cc -l static=l1 -l link-arg:+verbatim=a1 -l static=l2 -l link-arg=a2 -l dylib=d1 -l link-arg=a3 --print link-args | $(CGREP) -e 'l1.*"a1".*l2.*-Wl,a2.*d1.*-Wl,a3'
	$(RUSTC) rs.rs -Z unstable-options -C linker-flavor=ld -l static=l1 -l link-arg=a1 -l static=l2 -l link-arg=a2 -l dylib=d1 -l link-arg=a3 --print link-args | $(CGREP) -e 'l1.*"a1".*l2.*"a2".*d1.*"a3"'
