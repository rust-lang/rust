
snap-stage1: stage1/bin/rustc$(X) stage1/lib/$(CFG_RUNTIME) \
	stage1/lib/$(CFG_RUSTLLVM) stage1/lib/$(CFG_STDLIB)
	$(S)src/etc/make-snapshot.py stage1

snap-stage2: stage2/bin/rustc$(X) stage2/lib/$(CFG_RUNTIME) \
	stage2/lib/$(CFG_RUSTLLVM) stage2/lib/$(CFG_STDLIB)
	$(S)src/etc/make-snapshot.py stage2

snap-stage3: stage3/bin/rustc$(X) stage3/lib/$(CFG_RUNTIME) \
	stage3/lib/$(CFG_RUSTLLVM) stage3/lib/$(CFG_STDLIB)
	$(S)src/etc/make-snapshot.py stage3

