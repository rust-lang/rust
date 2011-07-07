
snap-stage1: stage1/rustc$(X) stage1/glue.o stage1/lib/$(CFG_STDLIB) \
	stage1/lib/libstd.rlib
	$(S)src/etc/make-snapshot.py stage1

snap-stage2: stage2/rustc$(X) stage2/glue.o stage2/lib/$(CFG_STDLIB) \
	stage2/lib/libstd.rlib
	$(S)src/etc/make-snapshot.py stage2

snap-stage3: stage3/rustc$(X) stage3/glue.o stage3/lib/$(CFG_STDLIB) \
	stage3/lib/libstd.rlib
	$(S)src/etc/make-snapshot.py stage3

