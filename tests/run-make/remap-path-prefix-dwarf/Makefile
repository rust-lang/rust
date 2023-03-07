# This test makes sure that --remap-path-prefix has the expected effects on paths in debuginfo.
# It tests several cases, each of them has a detailed description attached to it.

# ignore-windows

SRC_DIR := $(abspath .)
SRC_DIR_PARENT := $(abspath ..)

include ../../run-make-fulldeps/tools.mk

all: \
  abs_input_outside_working_dir \
  rel_input_remap_working_dir \
  rel_input_remap_working_dir_parent \
  rel_input_remap_working_dir_child \
  abs_input_inside_working_dir \
  abs_input_outside_working_dir

# The compiler is called with an *ABSOLUTE PATH* as input, and that absolute path *is* within
# the working directory of the compiler. We are remapping the path that contains `src`.
abs_input_inside_working_dir:
	# We explicitly switch to a directory that *is* a prefix of the directory our
	# source code is contained in.
	cd $(SRC_DIR) && $(RUSTC) $(SRC_DIR)/src/quux.rs -o "$(TMPDIR)/abs_input_inside_working_dir.rlib" -Cdebuginfo=2 --remap-path-prefix $(SRC_DIR)=REMAPPED
	# We expect the path to the main source file to be remapped.
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump $(TMPDIR)/abs_input_inside_working_dir.rlib | $(CGREP) "REMAPPED/src/quux.rs"
	# No weird duplication of remapped components (see #78479)
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump $(TMPDIR)/abs_input_inside_working_dir.rlib | $(CGREP) -v "REMAPPED/REMAPPED"

# The compiler is called with an *ABSOLUTE PATH* as input, and that absolute path is *not* within
# the working directory of the compiler. We are remapping both the path that contains `src` and
# the working directory to the same thing. This setup corresponds to a workaround that is needed
# when trying to remap everything to something that looks like a local path.
# Relative paths are interpreted as relative to the compiler's working directory (e.g. in
# debuginfo). If we also remap the working directory, the compiler strip it from other paths so
# that the final outcome is the desired one again.
abs_input_outside_working_dir:
	# We explicitly switch to a directory that is *not* a prefix of the directory our
	# source code is contained in.
	cd $(TMPDIR) && $(RUSTC) $(SRC_DIR)/src/quux.rs -o "$(TMPDIR)/abs_input_outside_working_dir.rlib" -Cdebuginfo=2 --remap-path-prefix $(SRC_DIR)=REMAPPED --remap-path-prefix $(TMPDIR)=REMAPPED
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump $(TMPDIR)/abs_input_outside_working_dir.rlib | $(CGREP) "REMAPPED/src/quux.rs"
	# No weird duplication of remapped components (see #78479)
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump $(TMPDIR)/abs_input_outside_working_dir.rlib | $(CGREP) -v "REMAPPED/REMAPPED"

# The compiler is called with a *RELATIVE PATH* as input. We are remapping the working directory of
# the compiler, which naturally is an implicit prefix of our relative input path. Debuginfo will
# expand the relative path to an absolute path and we expect the working directory to be remapped
# in that expansion.
rel_input_remap_working_dir:
	cd $(SRC_DIR) && $(RUSTC) src/quux.rs -o "$(TMPDIR)/rel_input_remap_working_dir.rlib" -Cdebuginfo=2 --remap-path-prefix "$(SRC_DIR)=REMAPPED"
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump "$(TMPDIR)/rel_input_remap_working_dir.rlib" | $(CGREP) "REMAPPED/src/quux.rs"
	# No weird duplication of remapped components (see #78479)
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump "$(TMPDIR)/rel_input_remap_working_dir.rlib" | $(CGREP) -v "REMAPPED/REMAPPED"

# The compiler is called with a *RELATIVE PATH* as input. We are remapping a *SUB-DIRECTORY* of the
# compiler's working directory. This test makes sure that that directory is remapped even though it
# won't actually show up in this form in the compiler's SourceMap and instead is only constructed
# on demand during debuginfo generation.
rel_input_remap_working_dir_child:
	cd $(SRC_DIR) && $(RUSTC) src/quux.rs -o "$(TMPDIR)/rel_input_remap_working_dir_child.rlib" -Cdebuginfo=2 --remap-path-prefix "$(SRC_DIR)/src=REMAPPED"
	# We expect `src/quux.rs` to have been remapped to `REMAPPED/quux.rs`.
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump "$(TMPDIR)/rel_input_remap_working_dir_child.rlib" | $(CGREP) "REMAPPED/quux.rs"
	# We don't want to find the path that we just remapped anywhere in the DWARF
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump "$(TMPDIR)/rel_input_remap_working_dir_child.rlib" | $(CGREP) -v "$(SRC_DIR)/src"
	# No weird duplication of remapped components (see #78479)
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump "$(TMPDIR)/rel_input_remap_working_dir_child.rlib" | $(CGREP) -v "REMAPPED/REMAPPED"

# The compiler is called with a *RELATIVE PATH* as input. We are remapping a *PARENT DIRECTORY* of
# the compiler's working directory.
rel_input_remap_working_dir_parent:
	cd $(SRC_DIR) && $(RUSTC) src/quux.rs -o "$(TMPDIR)/rel_input_remap_working_dir_parent.rlib" -Cdebuginfo=2 --remap-path-prefix "$(SRC_DIR_PARENT)=REMAPPED"
	# We expect `src/quux.rs` to have been remapped to `REMAPPED/remap-path-prefix-dwarf/src/quux.rs`.
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump "$(TMPDIR)/rel_input_remap_working_dir_parent.rlib" | $(CGREP) "REMAPPED/remap-path-prefix-dwarf/src/quux.rs"
	# We don't want to find the path that we just remapped anywhere in the DWARF
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump "$(TMPDIR)/rel_input_remap_working_dir_parent.rlib" | $(CGREP) -v "$(SRC_DIR_PARENT)"
	# No weird duplication of remapped components (see #78479)
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump "$(TMPDIR)/rel_input_remap_working_dir_parent.rlib" | $(CGREP) -v "REMAPPED/REMAPPED"
