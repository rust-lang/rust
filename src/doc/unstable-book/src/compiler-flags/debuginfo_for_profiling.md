# `debuginfo-for-profiling`

---

Emit extra debug info (currently it's [DWARF descriminators](https://llvm.org/doxygen/AddDiscriminators_8cpp.html)) to make a
sample profile more accurate. This flag is often used when creating a profile
for use with `-Cprofile-sample-use`.
