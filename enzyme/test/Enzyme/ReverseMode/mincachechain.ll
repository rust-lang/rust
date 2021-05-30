; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -adce -early-cse -S | FileCheck %s
; ModuleID = 'inp.ll'

declare dso_local void @_Z17__enzyme_autodiffPvPdS0_i(i8*, double*, double*) local_unnamed_addr #4
define dso_local void @outer(double* %m, double* %m2) local_unnamed_addr #2 {
entry:
  call void @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (void (double*)* @_Z10reduce_maxPdi to i8*), double* nonnull %m, double* nonnull %m2)
  ret void
}
; Function Attrs: nounwind uwtable
define dso_local void @_Z10reduce_maxPdi(double* %vec) #0 {
entry:
  %v = call double* @pb(double* %vec)
  call void @noop(double* %v)
  ret void
}

define double* @pb(double* %__x) {
entry:
  %a11 = call i64 @out(double* nonnull %__x)
  %a13 = call i64 @out(double* nonnull %__x)
  %sub = sub i64 %a11, %a13
  %s2 = add i64 %sub, 2
  %add.ptr.i = getelementptr inbounds double, double* %__x, i64 %s2
  call void @mid(double* %add.ptr.i)
  ret double* %__x
}

define void @mid(double* %mid) {
entry:
  %ld = load double, double* %mid, align 8
  %next = fadd double %ld, 1.000000e+00
  store double %next, double* %mid, align 8
  ret void
}

define i64 @out(double* %mid) {
entry:
  %int = ptrtoint double* %mid to i64
  ret i64 %int
}


define void @noop(double* %mid) {
entry:
  ret void
}


; CHECK: define internal void @diffe_Z10reduce_maxPdi(double* %vec, double* %"vec'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v_augmented = call { { i64, i64, i64 }, double*, double* } @augmented_pb(double* %vec, double* %"vec'")
; CHECK-NEXT:   %subcache = extractvalue { { i64, i64, i64 }, double*, double* } %v_augmented, 0
; CHECK-NEXT:   %v = extractvalue { { i64, i64, i64 }, double*, double* } %v_augmented, 1
; CHECK-NEXT:   %"v'ac" = extractvalue { { i64, i64, i64 }, double*, double* } %v_augmented, 2
; CHECK-NEXT:   call void @diffenoop(double* %v, double* %"v'ac")
; CHECK-NEXT:   call void @diffepb(double* %vec, double* %"vec'", { i64, i64, i64 } %subcache)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @augmented_mid(double* %mid, double* %"mid'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ld = load double, double* %mid, align 8
; CHECK-NEXT:   %next = fadd double %ld, 1.000000e+00
; CHECK-NEXT:   store double %next, double* %mid, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { i64, i64 } @augmented_out(double* %mid, double* %"mid'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"int'ipc" = ptrtoint double* %"mid'" to i64
; CHECK-NEXT:   %int = ptrtoint double* %mid to i64
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i64, i64 } undef, i64 %int, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i64, i64 } %.fca.0.insert, i64 %"int'ipc", 1
; CHECK-NEXT:   ret { i64, i64 } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { { i64, i64, i64 }, double*, double* } @augmented_pb(double* %__x, double* %"__x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a11_augmented = call { i64, i64 } @augmented_out(double* %__x, double* %"__x'")
; CHECK-NEXT:   %a11 = extractvalue { i64, i64 } %a11_augmented, 0
; CHECK-NEXT:   %"a11'ac" = extractvalue { i64, i64 } %a11_augmented, 1
; CHECK-NEXT:   %a13_augmented = call { i64, i64 } @augmented_out(double* %__x, double* %"__x'")
; CHECK-NEXT:   %a13 = extractvalue { i64, i64 } %a13_augmented, 0
; CHECK-NEXT:   %"a13'ac" = extractvalue { i64, i64 } %a13_augmented, 1
; CHECK-NEXT:   %sub = sub i64 %a11, %a13
; CHECK-NEXT:   %s2 = add i64 %sub, 2
; CHECK-NEXT:   %"add.ptr.i'ipg" = getelementptr inbounds double, double* %"__x'", i64 %s2
; CHECK-NEXT:   %add.ptr.i = getelementptr inbounds double, double* %__x, i64 %s2
; CHECK-NEXT:   call void @augmented_mid(double* %add.ptr.i, double* %"add.ptr.i'ipg")
; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { i64, i64, i64 }, double*, double* } undef, i64 %"a13'ac", 0, 0
; CHECK-NEXT:   %.fca.0.1.insert = insertvalue { { i64, i64, i64 }, double*, double* } %.fca.0.0.insert, i64 %"a11'ac", 0, 1
; CHECK-NEXT:   %.fca.0.2.insert = insertvalue { { i64, i64, i64 }, double*, double* } %.fca.0.1.insert, i64 %s2, 0, 2
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { i64, i64, i64 }, double*, double* } %.fca.0.2.insert, double* %__x, 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { { i64, i64, i64 }, double*, double* } %.fca.1.insert, double* %"__x'", 2
; CHECK-NEXT:   ret { { i64, i64, i64 }, double*, double* } %.fca.2.insert
; CHECK-NEXT: }

; CHECK: define internal void @diffepb(double* %__x, double* %"__x'", { i64, i64, i64 } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %s2 = extractvalue { i64, i64, i64 } %tapeArg, 2
; CHECK-NEXT:   %"add.ptr.i'ipg" = getelementptr inbounds double, double* %"__x'", i64 %s2
; CHECK-NEXT:   %add.ptr.i = getelementptr inbounds double, double* %__x, i64 %s2
; CHECK-NEXT:   call void @diffemid(double* %add.ptr.i, double* %"add.ptr.i'ipg")
; CHECK-NEXT:   call void @diffeout(double* %__x, double* %"__x'")
; CHECK-NEXT:   call void @diffeout(double* %__x, double* %"__x'")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffemid(double* %mid, double* %"mid'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"mid'", align 8
; CHECK-NEXT:   store double %0, double* %"mid'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffeout(double* %mid, double* %"mid'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
