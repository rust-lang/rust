; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

; Function Attrs: norecurse nounwind uwtable
define void @derivative(i32* %mat, i32* %dmat) {
entry:
  %call11 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (i32*)* @called to i8*), metadata !"enzyme_dup", i32* %mat, i32* %dmat)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local double @called(i32* %z) {
entry:
  %res = alloca double, align 8
  %call6 = call i64 @zz(i32* %z)
  %fp = uitofp i64 %call6 to double 
  %mat = bitcast i32* %z to double*
  %mat1 = getelementptr inbounds double, double* %mat, i64 1
  %call17 = call double @identity(double* %mat1)
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

; CHECK: define internal void @diffecalled(i32* %z, i32* %"z'", double %differeturn)
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   %call6_augmented = call { { { { {}, {}, i8*, i8* } } }, i64 } @augmented_zz(i32* %z, i32* %"z'")
; CHECK-ENZYME:   %call6 = extractvalue { { { { {}, {}, i8*, i8* } } }, i64 } %call6_augmented, 1
; CHECK-ENZYME:   %fp_unwrap = uitofp i64 %call6 to double
; CHECK-ENZYME:   %m0diffecall17 = fmul fast double %differeturn, %fp_unwrap
; CHECK-ENZYME:   %mat_unwrap = bitcast i32* %z to double*
; CHECK-ENZYME:   %mat1_unwrap = getelementptr inbounds double, double* %mat_unwrap, i64 1
; CHECK-ENZYME:   %"mat'ipc_unwrap" = bitcast i32* %"z'" to double*
; CHECK-ENZYME:   %"mat1'ipg_unwrap" = getelementptr inbounds double, double* %"mat'ipc_unwrap", i64 1
; CHECK-ENZYME:   call void @diffeidentity(double* %mat1_unwrap, double* %"mat1'ipg_unwrap", double %m0diffecall17)
; CHECK-ENZYME:   %_unwrap = extractvalue { { { { {}, {}, i8*, i8* } } }, i64 } %call6_augmented, 0
; CHECK-ENZYME:   call void @diffezz(i32* %z, i32* %"z'", { { { {}, {}, i8*, i8* } } } %_unwrap)
; CHECK-ENZYME:   ret void
; CHECK-ENZYME: }

; CHECK: define internal void @diffeidentity(double* %x, double* %"x'", double %differeturn)
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   %0 = load double, double* %"x'"
; CHECK-ENZYME:   %1 = fadd fast double %0, %differeturn
; CHECK-ENZYME:   store double %1, double* %"x'"
; CHECK-ENZYME:   ret void
; CHECK-ENZYME: }

; CHECK: define internal i64 @augmented_cols(i64* %this, i64* %"this'")
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   %a0 = load i64, i64* %this, align 8, !tbaa !0
; CHECK-ENZYME:   ret i64 %a0
; CHECK-ENZYME: }

; CHECK: define internal { i8*, i8* } @augmented_cast(i32* %this, i32* %"this'")
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   %"'ipc" = bitcast i32* %"this'" to i8*
; CHECK-ENZYME:   %0 = bitcast i32* %this to i8*
; CHECK-ENZYME:   %.fca.0.insert = insertvalue { i8*, i8* } undef, i8* %0, 0
; CHECK-ENZYME:   %.fca.1.insert = insertvalue { i8*, i8* } %.fca.1.insert, i8* %"'ipc", 1
; CHECK-ENZYME:   ret { i8*, i8* } %.fca.1.insert
; CHECK-ENZYME: }

; CHECK: define internal { { i8*, i64* }, i64 } @augmented_foo(i32* %this, i32* %"this'")
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   %call_augmented = call { i8*, i8* } @augmented_cast(i32* %this, i32* %"this'")
; CHECK-ENZYME:   %antiptr_call = extractvalue { i8*, i8* } %call_augmented, 1
; CHECK-ENZYME:   %call = extractvalue { i8*, i8* } %call_augmented, 0
; CHECK-ENZYME:   %"'ipc" = bitcast i8* %antiptr_call to i64*
; CHECK-ENZYME:   %0 = bitcast i8* %call to i64*
; CHECK-ENZYME:   %call2 = call i64 @augmented_cols(i64* %0, i64* %"'ipc")
; CHECK-ENZYME:   %.fca.0.0.insert = insertvalue { { i8*, i64* }, i64 } undef, i8* %antiptr_call, 0, 0
; CHECK-ENZYME:   %.fca.0.1.insert = insertvalue { { i8*, i64* }, i64 } %.fca.0.0.insert, i8* %0, 0, 1
; CHECK-ENZYME:   %.fca.1.insert = insertvalue { { i8*, i64* }, i64 } %.fca.0.1.insert, i64 %call2, 1
; CHECK-ENZYME:   ret { { i8*, i64* }, i64 } %.fca.1.insert
; CHECK-ENZYME: }

; CHECK: define internal { { i8*, i64* }, i64 } @augmented_sub(i32* %this, i32* %"this'")
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   %call_augmented = call { { i8*, i64* }, i64 } @augmented_foo(i32* %this, i32* %"this'")
; CHECK-ENZYME:   %subcache = extractvalue { { i8*, i64* }, i64 } %call_augmented, 0
; CHECK-ENZYME:   %subcache.fca.0.extract = extractvalue { i8*, i8* } %subcache, 0
; CHECK-ENZYME:   %subcache.fca.1.extract = extractvalue { i8*, i64* } %subcache, 1
; CHECK-ENZYME:   %call = extractvalue { { i8*, i64* }, i64 } %call_augmented, 1
; CHECK-ENZYME:   %.fca.0.0.insert = insertvalue { { i8*, i64* }, i64 } undef, i8* %subcache.fca.0.extract, 0, 0
; CHECK-ENZYME:   %.fca.0.1.insert = insertvalue { { i8*, i64* }, i64 } %.fca.0.0.insert, i8* %subcache.fca.1.extract, 0, 1
; CHECK-ENZYME:   %.fca.1.insert = insertvalue { { i8*, i64* }, i64 } %.fca.0.1.insert, i64 %call, 1
; CHECK-ENZYME:   ret { { i8*, i64* } }, i64 } %.fca.1.insert
; CHECK-ENZYME: }

; CHECK: define internal { { i8*, i64* }, i64 } @augmented_zz(i32* %this, i32* %"this'")
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   %call_augmented = call { { i8*, i64* }, i64 } @augmented_sub(i32* %this, i32* %"this'")
; CHECK-ENZYME:   %subcache = extractvalue { { i8*, i64* }, i64 } %call_augmented, 0
; CHECK-ENZYME:   %subcache.fca.0.extract = extractvalue { i8*, i64* } %subcache, 0
; CHECK-ENZYME:   %subcache.fca.1.extract = extractvalue { i8*, i64* } %subcache, 1
; CHECK-ENZYME:   %call = extractvalue { { { i8*, i64* } }, i64 } %call_augmented, 1
; CHECK-ENZYME:   %.fca.0.0.insert = insertvalue { { i8*, i64* }, i64 } undef, i8* %subcache.fca.0.extract, 0, 0
; CHECK-ENZYME:   %.fca.0.1.insert = insertvalue { { i8*, i64* }, i64 } %.fca.0.0.insert, i8* %subcache.fca.1.extract, 0, 1
; CHECK-ENZYME:   %.fca.1.insert = insertvalue { { i8*, i64* }, i64 } %.fca.0.1.insert, i64 %call, 1
; CHECK-ENZYME:   ret { { i8*, i64* }, i64 } %.fca.1.insert
; CHECK-ENZYME: }

; CHECK: define internal void @diffezz(i32* %this, i32* %"this'", { i8*, i64* } %[[tapeArg:.+]])
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   call void @diffesub(i32* %this, i32* %"this'", { i8*, i64* } %[[tapeArg]])
; CHECK-ENZYME:   ret void
; CHECK-ENZYME: }

; CHECK: define internal void @diffesub(i32* %this, i32* %"this'", { i8*, i64* } %[[tapeArg1:.+]])
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   call void @diffefoo(i32* %this, i32* %"this'", { i8*, i64* } %[[tapeArg1]])
; CHECK-ENZYME:   ret void
; CHECK-ENZYME: }

; CHECK: define internal void @diffefoo(i32* %this, i32* %"this'", { i8*, i64* } %[[tapeArg2:.+]])
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   %[[unwrap:.+]] = extractvalue { i8*, i8* } %[[tapeArg2]], 1
; CHECK-ENZYME:   %"call'ip_phi_fromtape_unwrap" = extractvalue { i8*, i8* } %[[tapeArg2]], 0
; CHECK-ENZYME:   %"'ipc_unwrap" = bitcast i8* %"call'ip_phi_fromtape_unwrap" to i64*
; CHECK-ENZYME:   call void @diffecols(i64* %[[unwrap]], i64* %"'ipc_unwrap")
; CHECK-ENZYME:   call void @diffecast(i32* %this, i32* %"this'")
; CHECK-ENZYME:   ret void
; CHECK-ENZYME: }

; CHECK: define internal void @diffecols(i64* %this, i64* %"this'")
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   ret void
; CHECK-ENZYME: }

; CHECK: define internal void @diffecast(i32* %this, i32* %"this'")
; CHECK-ENZYME: entry:
; CHECK-ENZYME:   ret void
; CHECK-ENZYME: }
