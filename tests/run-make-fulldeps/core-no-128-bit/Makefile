include ../tools.mk

all:
	$(RUSTC) --edition=2021 --crate-type=rlib --crate-name=core ../../../library/core/src/lib.rs --cfg no_128_bit
	$(RUSTC) --edition=2021 --crate-type=staticlib --crate-name=demo --sysroot=. -C panic=abort lib.rs
	# Expect that objdump succeeds and grep fails. The grep pattern is for names like __multi3.
	# There is no pipefail on dash so echo a string that will fail the test if objump fails.
	(objdump -t $(TMPDIR)/libdemo.a || echo __objdumpfailedti) | (! grep -w '__[a-z]\+ti[0-9]\?')
