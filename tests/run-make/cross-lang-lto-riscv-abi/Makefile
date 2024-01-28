# needs-matching-clang

# This test makes sure that cross-language LTO works on riscv targets

include ../tools.mk

all: riscv64gc-unknown-linux-gnu riscv32imac-unknown-none-elf riscv32gc-unknown-linux-gnu

define check-target =
@echo "Testing target $(1)"
$(RUSTC) --target $(1) -Clinker-plugin-lto=on -Cpanic=abort --crate-type=rlib -o $(TMPDIR)/libriscv-xlto.a ./riscv-xlto.rs
$(CLANG) -target $(2) -march=$(3) -mabi=$(4) -flto=thin -fuse-ld=lld -L $(TMPDIR) -lriscv-xlto -nostdlib -o $(TMPDIR)/riscv-xlto ./cstart.c
file $(TMPDIR)/riscv-xlto | $(CGREP) "$(5)"
endef


riscv64gc-unknown-linux-gnu:
	@$(call check-target,$@,riscv64-linux-gnu,rv64gc,lp64d,double-float ABI)

riscv32imac-unknown-none-elf:
	@$(call check-target,$@,riscv32-unknown-elf,rv32imac,ilp32,soft-float ABI)

riscv32gc-unknown-linux-gnu:
	@$(call check-target,$@,riscv32-linux-gnu,rv32gc,ilp32d,double-float ABI)
