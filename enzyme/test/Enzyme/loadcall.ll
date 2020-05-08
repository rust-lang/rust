; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -S | FileCheck %s
; ensure we don't emit a load of undef from the deleted call

source_filename = "loadcall.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local double* @gep(double* %a) #2 {
entry:
  ret double* %a
}

; Function Attrs: nounwind uwtable
define dso_local double @bad(double* %a) #0 {
entry:
  %call = call double* @gep(double* %a)
  %loaded = load double, double* %call, align 8, !tbaa !6
  ret double %loaded
}

; Function Attrs: nounwind uwtable
define dso_local double @meta(double (double*)* %f, double* %inp) #0 {
entry:
  %call = call fast double %f(double* %inp)
  ret double %call
}

; Function Attrs: nounwind uwtable
define dso_local double @fn(double* %a) #0 {
entry:
  %call = call fast double @meta(double (double*)* @bad, double* %a)
  ret double %call
}

; Function Attrs: nounwind uwtable
define dso_local void @caller(double* %vec, double* %dvec) #0 {
entry:
  %ad = call double (...) @__enzyme_autodiff.f64(double (double*)* @fn, double* %vec, double* %dvec)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #3

declare double @__enzyme_autodiff.f64(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline readnone nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"float", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !4, i64 0}

; CHECK: define internal {} @diffebad(double* %a, double* %"a'", double %differeturn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { {}, double* }*
; CHECK-NEXT:   %1 = load { {}, double* }, { {}, double* }* %0, !enzyme_mustcache !6
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %"loaded'de" = alloca double
; CHECK-NEXT:   store double 0.000000e+00, double* %"loaded'de"
; CHECK-NEXT:   %"call'ip_phi" = extractvalue { {}, double* } %1, 1
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double %differeturn, double* %"loaded'de"
; CHECK-NEXT:   %[[loadde:.+]] = load double, double* %"loaded'de"
; CHECK-NEXT:   store double 0.000000e+00, double* %"loaded'de"
; CHECK-NEXT:   %"call'ip_phi_fromtape_unwrap" = extractvalue { {}, double* } %1, 1
; CHECK-NEXT:   %[[ligep:.+]] = load double, double* %"call'ip_phi_fromtape_unwrap", align 8
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %[[ligep]], %[[loadde]]
; CHECK-NEXT:   store double %[[add]], double* %"call'ip_phi_fromtape_unwrap", align 8
; CHECK-NEXT:   %{{.+}} = call {} @diffegep(double* %a, double* %"a'", {} undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
