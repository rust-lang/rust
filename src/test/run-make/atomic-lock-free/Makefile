-include ../tools.mk

# This tests ensure that atomic types are never lowered into runtime library calls that are not
# guaranteed to be lock-free.

all:
ifeq ($(UNAME),Linux)
	$(RUSTC) --target=i686-unknown-linux-gnu atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=x86_64-unknown-linux-gnu atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=arm-unknown-linux-gnueabi atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=arm-unknown-linux-gnueabihf atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=armv7-unknown-linux-gnueabihf atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=aarch64-unknown-linux-gnu atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=mips-unknown-linux-gnu atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=mipsel-unknown-linux-gnu atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=powerpc-unknown-linux-gnu atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=powerpc64-unknown-linux-gnu atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
	$(RUSTC) --target=powerpc64le-unknown-linux-gnu atomic_lock_free.rs
	nm "$(TMPDIR)/libatomic_lock_free.rlib" | grep -vq __atomic_fetch_add
endif
