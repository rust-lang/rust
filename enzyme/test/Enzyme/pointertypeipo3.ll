; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s
; XFAIL: *

declare void @__enzyme_autodiff(...)

; Function Attrs: alwaysinline norecurse nounwind uwtable
define dso_local double @caller() local_unnamed_addr #0 {
entry:
  %kernel = alloca i64, align 8
  %kernelp = alloca i64, align 8
  %akernel = alloca i64, align 8
  %akernelp = alloca i64, align 8
  call void (...) @__enzyme_autodiff(void (i64*, i64*)* nonnull @mv, metadata !"diffe_dup", i64* nonnull %kernel, i64* nonnull %kernelp, metadata !"diffe_dup", i64* nonnull %akernel, i64* nonnull %akernelp) #4
  ret double 0.000000e+00
}

; Function Attrs: alwaysinline nounwind uwtable
define internal void @mv(i64* %m_dims, i64* %out) #1 {
entry:
  %call4 = call i64 @sub(i64* nonnull %m_dims)
  store i64 %call4, i64* %out
  ret void
}

define i64 @mul(i64 %a) {
entry:
  ret i64 %a
}

; Function Attrs: inlinehint norecurse nounwind readnone uwtable
define i64 @pop(i64 %arr.coerce0)  {
entry:
  %arr = alloca i64
  store i64 %arr.coerce0, i64* %arr
  %call.i = call i64* @cast(i64* nonnull %arr)
  %a2 = load i64, i64* %call.i, !tbaa !2
  %call2 = call i64 @mul(i64 %a2)
  ret i64 %call2
}

define i64* @cast(i64* %a) {
entry:
  ret i64* %a
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define i64 @sub(i64* %this) {
entry:
  %agg = load i64, i64* %this
  %call = tail call i64 @pop(i64 %agg)
  ret i64 %call
}

attributes #0 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"double"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: define internal {} @diffemv(i64* %m_dims, i64* %"m_dims'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {} @diffesub(i64* %m_dims, i64* %"m_dims'")
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffesub(i64* %this, i64* %"this'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg = load i64, i64* %this, align 4
; CHECK-NEXT:   %call = tail call i64 @pop(i64 %agg)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
