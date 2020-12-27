; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare void @__enzyme_autodiff(...)

declare dso_local void @abort(i64 %parm)

define internal double @f(double** %xp, i64* %lenp) {
top:
  %len = load i64, i64* %lenp, align 8
  %cmp = icmp eq i64 %len, 0
  br i1 %cmp, label %oob, label %idxend2

oob:                                              ; preds = %top
  call void @abort(i64 %len)
  unreachable

idxend2:                                          ; preds = %top
  %x = load double*, double** %xp
  %val = load double, double* %x  
  %res = fmul double %val, %val
  ret double %res
}

define internal double @g(double** %xp, i64* %lenp) {
entry:
  %res = call double @f(double** %xp, i64* %lenp)
  store i64 0, i64* %lenp
  store double* null, double** %xp
  ret double %res 
}

define double @caller(double** %xp, double** %d_xp, i64* %lenp) local_unnamed_addr #0 {
entry:
  call void (...) @__enzyme_autodiff(double (double**, i64*)* nonnull @g, double** %xp, double** %d_xp, i64* %lenp)
  ret double 0.000000e+00
}

; TODO forgo caching double* %x, since it isn't going to be used [needing the cache of %val]
; CHECK: define internal { double, double*, double* } @augmented_f(double** %xp, double** %"xp'", i64* %lenp)
; CHECK-NEXT: top:
; CHECK-NEXT:   %len = load i64, i64* %lenp, align 8
; CHECK-NEXT:   %cmp = icmp eq i64 %len, 0
; CHECK-NEXT:   br i1 %cmp, label %oob, label %idxend2

; CHECK: oob:                                              ; preds = %top
; CHECK-NEXT:   call void @abort(i64 %len)
; CHECK-NEXT:   unreachable

; CHECK: idxend2:                                          ; preds = %top
; CHECK-NEXT:   %"x'ipl" = load double*, double** %"xp'"
; CHECK-NEXT:   %x = load double*, double** %xp
; CHECK-NEXT:   %val = load double, double* %x
; CHECK-NEXT:   %.fca.0.insert = insertvalue { double, double*, double* } undef, double %val, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { double, double*, double* } %.fca.0.insert, double* %"x'ipl", 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { double, double*, double* } %.fca.1.insert, double* %x, 2
; CHECK-NEXT:   ret { double, double*, double* } %.fca.2.insert
; CHECK-NEXT: }

; CHECK: define internal void @diffef(double** %xp, double** %"xp'", i64* %lenp, double %differeturn, { double, double*, double* } %tapeArg)
; CHECK-NEXT: top:
; CHECK-NEXT:   %"x'il_phi" = extractvalue { double, double*, double* } %tapeArg, 1
; CHECK-NEXT:   %val = extractvalue { double, double*, double* } %tapeArg, 0
; CHECK-NEXT:   %m0diffeval = fmul fast double %differeturn, %val
; CHECK-NEXT:   %m1diffeval = fmul fast double %differeturn, %val
; CHECK-NEXT:   %0 = fadd fast double %m0diffeval, %m1diffeval
; CHECK-NEXT:   %1 = load double, double* %"x'il_phi"
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   store double %2, double* %"x'il_phi"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }