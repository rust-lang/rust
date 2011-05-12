
snap-stage1: stage1/rustc stage1/glue.o stage1/$(CFG_STDLIB)
	$(CFG_SRC_DIR)src/etc/make-snapshot.py stage1

snap-stage2: stage2/rustc stage2/glue.o stage2/$(CFG_STDLIB)
	$(CFG_SRC_DIR)src/etc/make-snapshot.py stage2

snap-stage3: stage3/rustc stage3/glue.o stage3/$(CFG_STDLIB)
	$(CFG_SRC_DIR)src/etc/make-snapshot.py stage3

