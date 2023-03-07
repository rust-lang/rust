; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

declare dso_local double @__enzyme_autodiff(...)

define double @square(double (double, double)* %add, double %x) {
entry:
  %mul = call double %add(double %x, double 1.0000000e+00)
  ret double %mul
}

define double @dsquare(double (double, double)* %add, double (double, double)* %dadd, double %x) local_unnamed_addr {
entry:
  %call = tail call double (...) @__enzyme_autodiff(i8* bitcast (double (double (double, double)*, double)* @square to i8*), 
    metadata !"enzyme_dup", double (double, double)* %add, double (double, double)* %dadd, double %x)
  ret double %call
}

; CHECK: define internal { double } @diffesquare(double (double, double)* %add, double (double, double)* %"add'", double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast double (double, double)* %add to i8*
; CHECK-NEXT:   %1 = bitcast double (double, double)* %"add'" to i8*
; CHECK-NEXT:   %2 = icmp eq i8* %0, %1
; CHECK-NEXT:   br i1 %2, label %error.i, label %__enzyme_runtimeinactiveerr.exit

; CHECK: error.i:                                          ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @puts(i8* getelementptr inbounds ([79 x i8], [79 x i8]* @.str, i32 0, i32 0))
; CHECK-NEXT:   call void @exit(i32 1)
; CHECK-NEXT:   unreachable

; CHECK: __enzyme_runtimeinactiveerr.exit:                 ; preds = %entry
; CHECK-NEXT:   %4 = bitcast double (double, double)* %"add'" to { i8*, double } (double, double)**
; CHECK-NEXT:   %5 = load { i8*, double } (double, double)*, { i8*, double } (double, double)** %4
; CHECK-NEXT:   %mul_augmented = call { i8*, double } %5(double %x, double 1.000000e+00)
; CHECK-NEXT:   %subcache = extractvalue { i8*, double } %mul_augmented, 0
; CHECK-NEXT:   %6 = bitcast double (double, double)* %"add'" to { double, double } (double, double, double, i8*)**
; CHECK-NEXT:   %7 = getelementptr { double, double } (double, double, double, i8*)*, { double, double } (double, double, double, i8*)** %6, i64 1
; CHECK-NEXT:   %8 = load { double, double } (double, double, double, i8*)*, { double, double } (double, double, double, i8*)** %7
; CHECK-NEXT:   %9 = call { double, double } %8(double %x, double 1.000000e+00, double %differeturn, i8* %subcache)
; CHECK-NEXT:   %10 = extractvalue { double, double } %9, 0
; CHECK-NEXT:   %11 = insertvalue { double } undef, double %10, 0
; CHECK-NEXT:   ret { double } %11
; CHECK-NEXT: }
