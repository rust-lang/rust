; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)
define void @derivative(i64* %this, i64* %dthis) {
entry:
  %call11 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (i64*)* @myfunc to i8*), metadata !"enzyme_dup", i64* %this, i64* %dthis)
  ret void
}

define linkonce_odr dso_local double @myfunc(i64* %this) {
entry:
  %call = tail call i64 @size(i64* %this)
  %cmp = icmp eq i64 %call, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi double [ 2.000000e+00, %if.end ], [ 0.000000e+00, %entry ]
  ret double %retval.0
}

define i64 @size(i64* %this) { 
entry:
  %call = tail call i64 @rows(i64* %this)
  %call2 = tail call i64 @ident(i64 %call)
  ret i64 %call2
}

define linkonce_odr dso_local i64 @ident(i64 %m_rows) {
entry:
  ret i64 %m_rows
}

define linkonce_odr dso_local i64 @rows(i64* %this) {
entry:
  %call = tail call i64 @loader(i64* %this)
  ret i64 %call
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local i64 @loader(i64* %m_rows) {
entry:
  %0 = load i64, i64* %m_rows, align 8, !tbaa !15
  ret i64 %0
}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!9 = !{!"long", !4, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!14 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !7, i64 0, !9, i64 8, !9, i64 16}
!15 = !{!14, !9, i64 16}

; CHECK: @diffemyfunc
