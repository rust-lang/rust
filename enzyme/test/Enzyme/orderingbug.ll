; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

; Function Attrs: norecurse nounwind uwtable
define void @derivative(double* %mat, double* %dmat) {
entry:
  %call11 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*)* @called to i8*), double* %mat, double* %dmat)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local double @called(double* %mat) {
entry:
  %res = alloca double, align 8
  %z = bitcast double* %mat to i32*
  %call6 = call i64 @zz(i32* %z)
  %fp = uitofp i64 %call6 to double   
  %call17 = call double @identity(double* %mat)
  %mul = fmul double %call17, %fp
  ret double %mul
}

define i64 @zz(i32* %this) { 
entry:
  %call = tail call i64 @sub(i32* %this)
  ret i64 %call
}

define i64 @sub(i32* %this) {
entry:
  %call = tail call i64 @foo(i32* %this)
  ret i64 %call
}

define i64 @foo(i32* %this) {
entry:
  %call = tail call i8* @cast(i32* %this)
  %0 = bitcast i8* %call to i64*
  %call2 = tail call i64 @cols(i64* %0)
  ret i64 %call2
}

define i8* @cast(i32* %this) {
entry:
  %0 = bitcast i32* %this to i8*
  ret i8* %0
}

define i64 @cols(i64* %this) {
entry:
  %a0 = load i64, i64* %this, align 8, !tbaa !15
  ret i64 %a0
}

define double @identity(double* %x) {
entry:
    %z = load double, double* %x
    ret double %z
}

!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!9 = !{!"long", !4, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!14 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !7, i64 0, !9, i64 8, !9, i64 16}
!15 = !{!14, !9, i64 16}
