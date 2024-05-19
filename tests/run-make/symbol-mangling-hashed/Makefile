include ../tools.mk

# ignore-cross-compile
# only-linux
# only-x86_64

NM=nm -D
RLIB_NAME=liba_rlib.rlib
DYLIB_NAME=liba_dylib.so
SO_NAME=libb_dylib.so
BIN_NAME=b_bin

ifeq ($(UNAME),Darwin)
NM=nm -gU
RLIB_NAME=liba_rlib.rlib
DYLIB_NAME=liba_dylib.dylib
SO_NAME=libb_dylib.dylib
BIN_NAME=b_bin
endif

ifdef IS_WINDOWS
NM=nm -g
RLIB_NAME=liba_rlib.dll.a
DYLIB_NAME=liba_dylib.dll
SO_NAME=libb_dylib.dll
BIN_NAME=b_bin.exe
endif

all:
	$(RUSTC) -C prefer-dynamic -Z unstable-options -C symbol-mangling-version=hashed -C metadata=foo a_dylib.rs
	$(RUSTC) -C prefer-dynamic -Z unstable-options -C symbol-mangling-version=hashed -C metadata=bar a_rlib.rs
	$(RUSTC) -C prefer-dynamic -L $(TMPDIR) b_dylib.rs
	$(RUSTC) -C prefer-dynamic -L $(TMPDIR) b_bin.rs

    # Check hashed symbol name

	[ "$$($(NM) $(TMPDIR)/$(DYLIB_NAME) | grep -c hello)" -eq "0" ]
	[ "$$($(NM) $(TMPDIR)/$(DYLIB_NAME) | grep _RNxC7a_dylib | grep -c ' T ')" -eq "1" ]

	[ "$$($(NM) $(TMPDIR)/$(SO_NAME) | grep b_dylib | grep -c hello)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(SO_NAME) | grep _RNxC6a_rlib | grep -c ' T ')" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(SO_NAME) | grep _RNxC7a_dylib | grep -c ' U ')" -eq "1" ]

	[ "$$($(NM) $(TMPDIR)/$(BIN_NAME) | grep _RNxC6a_rlib | grep -c ' U ')" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(BIN_NAME) | grep _RNxC7a_dylib | grep -c ' U ')" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(BIN_NAME) | grep b_dylib | grep hello | grep -c ' U ')" -eq "1" ]

	$(call RUN,$(BIN_NAME))
