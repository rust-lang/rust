; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -early-cse -adce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { double*, i64, i64 }

define double @caller(%"class.Eigen::Matrix"* %A, %"class.Eigen::Matrix"* %Ap) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (%"class.Eigen::Matrix"*)* @matvec to i8*), %"class.Eigen::Matrix"* %A, %"class.Eigen::Matrix"* %Ap)
  ret double %call
}

declare double @__enzyme_autodiff(i8*, ...)

; Function Attrs: noinline nounwind uwtable
define internal void @matvec(%"class.Eigen::Matrix"* noalias %W) #0 {
entry:
  %m_rows.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
  %z0 = load i64, i64* %m_rows.i.i.i.i.i.i.i.i.i, align 8, !tbaa !2
  %mul.i.i.i.i = shl i64 %z0, 3
  %call.i.i4.i.i.i.i = call noalias i8* @malloc(i64 %mul.i.i.i.i) #2
  %res = bitcast i8* %call.i.i4.i.i.i.i to double*
  %div.i.i.i.i = sdiv i64 %z0, 2
  %mul.i.i.i.i1 = shl nsw i64 %div.i.i.i.i, 1
  %z2 = bitcast i8* %call.i.i4.i.i.i.i to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %z2, align 16, !tbaa !8
  %arrayidx.i.i.i.i.i.i.i = getelementptr inbounds double, double* %res, i64 %mul.i.i.i.i1
  %z3 = bitcast double* %arrayidx.i.i.i.i.i.i.i to i64*
  store i64 0, i64* %z3, align 8, !tbaa !9
  %lhs = bitcast %"class.Eigen::Matrix"* %W to i64*
  call void @subfn(i64* %lhs, double* %res, i1 true) #2
  call void @free(i8* %call.i.i4.i.i.i.i) #2
  ret void
}

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @subfn(i64* %lhs, double* %argres, i1 %cmp.i.i.i) local_unnamed_addr #0 {
entry:
  %a0 = ptrtoint double* %argres to i64
  %a2 = load i64, i64* %lhs, align 8
  %a3 = inttoptr i64 %a2 to double*
  %cond.i.i.i = select i1 %cmp.i.i.i, i64 %a2, i64 0
  %cmp17 = icmp eq i64 %cond.i.i.i, %a0
  %idx = zext i1 %cmp17 to i64
  %arrayidx.i.i814 = getelementptr inbounds double, double* %a3, i64 %idx
  %a4 = bitcast double* %arrayidx.i.i814 to i64*
  %a51 = load i64, i64* %a4, align 8, !tbaa !10
  %a5 = bitcast double* %argres to i64*
  store i64 %a51, i64* %a5, align 8
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !7, i64 8}
!3 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !4, i64 0, !7, i64 8, !7, i64 16}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!5, !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !5, i64 0}

; CHECK: define internal void @diffematvec(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix"* %"W'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m_rows.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
; CHECK-NEXT:   %z0 = load i64, i64* %m_rows.i.i.i.i.i.i.i.i.i, align 8, !tbaa !2
; CHECK-NEXT:   %mul.i.i.i.i = shl i64 %z0, 3
; CHECK-NEXT:   %call.i.i4.i.i.i.i = call noalias i8* @malloc(i64 %mul.i.i.i.i)
; CHECK-NEXT:   %"call.i.i4.i.i.i.i'mi" = call noalias nonnull i8* @malloc(i64 %mul.i.i.i.i)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"call.i.i4.i.i.i.i'mi", i8 0, i64 %mul.i.i.i.i, i1 false)
; CHECK-NEXT:   %[[resipc:.+]] = bitcast i8* %"call.i.i4.i.i.i.i'mi" to double*
; CHECK-NEXT:   %res = bitcast i8* %call.i.i4.i.i.i.i to double*
; CHECK-NEXT:   %div.i.i.i.i = sdiv i64 %z0, 2
; CHECK-NEXT:   %mul.i.i.i.i1 = shl nsw i64 %div.i.i.i.i, 1
; CHECK-NEXT:   %[[z2ipc:.+]] = bitcast i8* %"call.i.i4.i.i.i.i'mi" to <2 x double>*
; CHECK-NEXT:   %z2 = bitcast i8* %call.i.i4.i.i.i.i to <2 x double>*
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %z2, align 16, !tbaa !8
; CHECK-NEXT:   %[[arrayidxipge:.+]] = getelementptr inbounds double, double* %[[resipc]], i64 %mul.i.i.i.i1
; CHECK-NEXT:   %arrayidx.i.i.i.i.i.i.i = getelementptr inbounds double, double* %res, i64 %mul.i.i.i.i1
; CHECK-NEXT:   %[[z3ipc:.+]] = bitcast double* %[[arrayidxipge]] to i64*
; CHECK-NEXT:   %z3 = bitcast double* %arrayidx.i.i.i.i.i.i.i to i64*
; CHECK-NEXT:   store i64 0, i64* %z3, align 8, !tbaa !9
; CHECK-NEXT:   %[[lhsipc:.+]] = bitcast %"class.Eigen::Matrix"* %"W'" to i64*
; CHECK-NEXT:   %lhs = bitcast %"class.Eigen::Matrix"* %W to i64*
; CHECK-NEXT:   call void @diffesubfn(i64* %lhs, i64* %[[lhsipc]], double* %res, double*{{( nonnull)?}} %[[resipc]], i1 true)
; CHECK-NEXT:   store i64 0, i64* %[[z3ipc]], align 8
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %[[z2ipc]], align 16
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call.i.i4.i.i.i.i'mi")
; CHECK-NEXT:   tail call void @free(i8* %call.i.i4.i.i.i.i)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesubfn(i64* %lhs, i64* %"lhs'", double* %argres, double* %"argres'", i1 %cmp.i.i.i)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a0 = ptrtoint double* %argres to i64
; CHECK-NEXT:   %[[a2p:.+]] = load i64, i64* %"lhs'", align 8
; CHECK-NEXT:   %a2 = load i64, i64* %lhs, align 8
; CHECK-NEXT:   %[[a3p:.+]] = inttoptr i64 %[[a2p]] to double*
; CHECK-NEXT:   %a3 = inttoptr i64 %a2 to double*
; CHECK-NEXT:   %cond.i.i.i = select i1 %cmp.i.i.i, i64 %a2, i64 0
; CHECK-NEXT:   %cmp17 = icmp eq i64 %cond.i.i.i, %a0
; CHECK-NEXT:   %idx = zext i1 %cmp17 to i64
; CHECK-NEXT:   %[[arrayidxi814ipge:.+]] = getelementptr inbounds double, double* %[[a3p]], i64 %idx
; CHECK-NEXT:   %arrayidx.i.i814 = getelementptr inbounds double, double* %a3, i64 %idx
; CHECK-NEXT:   %a4 = bitcast double* %arrayidx.i.i814 to i64*
; CHECK-NEXT:   %a51 = load i64, i64* %a4, align 8
; CHECK-NEXT:   %[[a5ipc:.+]] = bitcast double* %"argres'" to i64*
; CHECK-NEXT:   %a5 = bitcast double* %argres to i64*
; CHECK-NEXT:   store i64 %a51, i64* %a5, align 8

; CHECK-NEXT:   %[[lres:.+]] = load i64, i64* %[[a5ipc]], align 8
; CHECK-NEXT:   store i64 0, i64* %[[a5ipc]], align 8

; CHECK-DAG:   %[[lresd:.+]] = bitcast i64 %[[lres]] to double
; CHECK-DAG:   %[[tloadd:.+]] = load double, double* %[[arrayidxi814ipge]]

; CHECK-DAG:   %[[fld:.+]] = fadd fast double %[[tloadd]], %[[lresd]]

; CHECK-NEXT:   store double %[[fld]], double* %[[arrayidxi814ipge]], align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

