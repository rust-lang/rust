include ../tools.mk

SRC=test.rs
OUT=$(TMPDIR)/out

all: asm llvm-ir dep-info mir llvm-bc obj metadata link multiple-types multiple-types-option-o

asm: $(OUT)
	$(RUSTC) --emit asm=$(OUT)/$@ $(SRC)
	$(RUSTC) --emit asm=- $(SRC) | diff - $(OUT)/$@
llvm-ir: $(OUT)
	$(RUSTC) --emit llvm-ir=$(OUT)/$@ $(SRC)
	$(RUSTC) --emit llvm-ir=- $(SRC) | diff - $(OUT)/$@
dep-info: $(OUT)
	$(RUSTC) -Z dep-info-omit-d-target=yes --emit dep-info=$(OUT)/$@ $(SRC)
	$(RUSTC) --emit dep-info=- $(SRC) | diff - $(OUT)/$@
mir: $(OUT)
	$(RUSTC) --emit mir=$(OUT)/$@ $(SRC)
	$(RUSTC) --emit mir=- $(SRC) | diff - $(OUT)/$@

llvm-bc: $(OUT)
	$(RUSTC) --emit llvm-bc=- $(SRC) 1>/dev/ptmx 2>$(OUT)/$@ || true
	diff $(OUT)/$@ emit-llvm-bc.stderr
obj: $(OUT)
	$(RUSTC) --emit obj=- $(SRC) 1>/dev/ptmx 2>$(OUT)/$@ || true
	diff $(OUT)/$@ emit-obj.stderr

# For metadata output, a temporary directory will be created to hold the temporary
# metadata file. But when output is stdout, the temporary directory will be located
# in the same place as $(SRC), which is mounted as read-only in the tests. Thus as
# a workaround, $(SRC) is copied to the test output directory $(OUT) and we compile
# it there.
metadata: $(OUT)
	cp $(SRC) $(OUT)
	(cd $(OUT); $(RUSTC) --emit metadata=- $(SRC) 1>/dev/ptmx 2>$(OUT)/$@ || true)
	diff $(OUT)/$@ emit-metadata.stderr

link: $(OUT)
	$(RUSTC) --emit link=- $(SRC) 1>/dev/ptmx 2>$(OUT)/$@ || true
	diff $(OUT)/$@ emit-link.stderr

multiple-types: $(OUT)
	$(RUSTC) --emit asm=- --emit llvm-ir=- --emit dep-info=- --emit mir=- $(SRC) 2>$(OUT)/$@ || true
	diff $(OUT)/$@ emit-multiple-types.stderr

multiple-types-option-o: $(OUT)
	$(RUSTC) -o - --emit asm,llvm-ir,dep-info,mir $(SRC) 2>$(OUT)/$@ || true
	diff $(OUT)/$@ emit-multiple-types.stderr

$(OUT):
	mkdir -p $(OUT)
