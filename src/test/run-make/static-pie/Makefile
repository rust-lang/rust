include ../../run-make-fulldeps/tools.mk

# only-x86_64
# only-linux
# ignore-gnux32

# How to manually run this
# $ ./x.py test --target x86_64-unknown-linux-[musl,gnu] src/test/run-make/static-pie

all: test-clang test-gcc

test-%:
	if ./check_$*_version.sh; then\
		${RUSTC} -Clinker=$* -Clinker-flavor=gcc --target ${TARGET} -C target-feature=+crt-static test-aslr.rs; \
		! readelf -l $(call RUN_BINFILE,test-aslr) | $(CGREP) INTERP; \
		readelf -l $(call RUN_BINFILE,test-aslr) | $(CGREP) DYNAMIC; \
		$(call RUN,test-aslr) --test-aslr; \
	fi
