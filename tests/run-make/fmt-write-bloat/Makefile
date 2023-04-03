include ../tools.mk

# ignore-windows

ifeq ($(shell $(RUSTC) -vV | grep 'host: $(TARGET)'),)

# Don't run this test when cross compiling.
all:

else

NM = nm

PANIC_SYMS = panic_bounds_check Debug

# Allow for debug_assert!() in debug builds of std.
ifdef NO_DEBUG_ASSERTIONS
PANIC_SYMS += panicking panic_fmt pad_integral Display Debug
endif

all: main.rs
	$(RUSTC) $< -O
	$(NM) $(call RUN_BINFILE,main) | $(CGREP) -v $(PANIC_SYMS)

endif
