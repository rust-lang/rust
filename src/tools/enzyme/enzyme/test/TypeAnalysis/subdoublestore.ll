; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=foo -o /dev/null | FileCheck %s

source_filename = "subdoublestore.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"



; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define void @foo(double %inp) #3 {
entry:
  %conv = bitcast double %inp to i64
  %call = tail call i64* @substore(i64 %conv, i64 3)
  %0 = bitcast i64* %call to double*
  %1 = load double, double* %0
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local noalias i64* @substore(i64 %flt, i64 %integral) local_unnamed_addr #1 {
entry:
  %call = tail call noalias i8* @malloc(i64 16) #6
  %0 = bitcast i8* %call to i64*
  store i64 %flt, i64* %0
  %arrayidx1 = getelementptr inbounds i8, i8* %call, i64 8
  %1 = bitcast i8* %arrayidx1 to i64*
  store i64 %integral, i64* %1
  ret i64* %0
}

declare dso_local double @__enzyme_autodiff(i8*, ...) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #5

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"long long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}

; CHECK: foo - {} |{[-1]:Float@double}:{}
; CHECK-NEXT: double %inp: {[-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %conv = bitcast double %inp to i64: {[-1]:Float@double}
; CHECK-NEXT:   %call = tail call i64* @substore(i64 %conv, i64 3): {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}
; CHECK-NEXT:   %0 = bitcast i64* %call to double*: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}
; CHECK-NEXT:   %1 = load double, double* %0{{(, align 8)?}}: {[-1]:Float@double}
; CHECK-NEXT:   ret void: {}

; CHECK: substore - {[-1]:Pointer} |{[-1]:Float@double}:{} {[-1]:Integer}:{3,}
; CHECK-NEXT: i64 %flt: {[-1]:Float@double}
; CHECK-NEXT: i64 %integral: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %call = tail call noalias i8* @malloc(i64 16) #5: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}
; CHECK-NEXT:   %0 = bitcast i8* %call to i64*: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}
; CHECK-NEXT:   store i64 %flt, i64* %0{{(, align 8)?}}: {}
; CHECK-NEXT:   %arrayidx1 = getelementptr inbounds i8, i8* %call, i64 8: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   %1 = bitcast i8* %arrayidx1 to i64*: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   store i64 %integral, i64* %1{{(, align 8)?}}: {}
; CHECK-NEXT:   ret i64* %0: {}
