
snap-stage1: $(HOST_BIN1)/rustc$(X) $(HOST_LIB1)/$(CFG_RUNTIME) \
	$(HOST_LIB1)/$(CFG_RUSTLLVM) $(HOST_LIB1)/$(CFG_STDLIB)
	$(S)src/etc/make-snapshot.py stage1

snap-stage1: $(HOST_BIN2)/rustc$(X) $(HOST_LIB2)/$(CFG_RUNTIME) \
	$(HOST_LIB2)/$(CFG_RUSTLLVM) $(HOST_LIB2)/$(CFG_STDLIB)
	$(S)src/etc/make-snapshot.py stage1

snap-stage1: $(HOST_BIN3)/rustc$(X) $(HOST_LIB3)/$(CFG_RUNTIME) \
	$(HOST_LIB3)/$(CFG_RUSTLLVM) $(HOST_LIB3)/$(CFG_STDLIB)
	$(S)src/etc/make-snapshot.py stage1
