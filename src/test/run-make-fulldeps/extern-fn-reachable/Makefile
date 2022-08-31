include ../tools.mk

# ignore-windows-msvc

NM=nm -D

ifeq ($(UNAME),Darwin)
NM=nm -gU
endif

ifdef IS_WINDOWS
NM=nm -g
endif

# This overrides the LD_LIBRARY_PATH for RUN
TARGET_RPATH_DIR:=$(TARGET_RPATH_DIR):$(TMPDIR)

all:
	$(RUSTC) dylib.rs -o $(TMPDIR)/libdylib.so -C prefer-dynamic

	[ "$$($(NM) $(TMPDIR)/libdylib.so | grep -v __imp_ | grep -c fun1)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/libdylib.so | grep -v __imp_ | grep -c fun2)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/libdylib.so | grep -v __imp_ | grep -c fun3)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/libdylib.so | grep -v __imp_ | grep -c fun4)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/libdylib.so | grep -v __imp_ | grep -c fun5)" -eq "1" ]
