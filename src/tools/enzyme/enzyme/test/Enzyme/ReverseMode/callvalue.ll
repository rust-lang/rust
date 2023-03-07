; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -S -early-cse -adce | FileCheck %s

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local double @square(double %x) #0 {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @indirect(double (double)* nocapture %callee, double %x) local_unnamed_addr #1 {
entry:
  %call = tail call fast double %callee(double %x) #2
  ret double %call
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @create(double %x) #1 {
entry:
  %call = tail call fast double @indirect(double (double)* nonnull @square, double %x)
  ret double %call
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @create, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...) #2

attributes #0 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}

; CHECK: @"_enzyme_reverse_square'" = internal constant { { i8*, double } (double)*, { double } (double, double, i8*)* } { { i8*, double } (double)* @augmented_square, { double } (double, double, i8*)* @diffesquare }

; CHECK: define internal { double } @diffecreate(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffeindirect(double (double)* @square, double (double)* bitcast ({ { i8*, double } (double)*, { double } (double, double, i8*)* }* @"_enzyme_reverse_square'" to double (double)*), double %x, double %differeturn)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { i8*, double } @augmented_square(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul fast double %x, %x
; CHECK-NEXT:   %[[iv2:.+]] = insertvalue { i8*, double } { i8* null, double {{(undef|poison)?}} }, double %mul, 1
; CHECK-NEXT:   ret { i8*, double } %[[iv2]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesquare(double %x, double %differeturn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %[[mul:.+]] = fmul fast double %differeturn, %x
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %[[mul]], %[[mul]]
; CHECK-NEXT:   %[[rv:.+]] = insertvalue { double } undef, double %[[add]], 0
; CHECK-NEXT:   ret { double } %[[rv]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeindirect(double (double)* nocapture %callee, double (double)* nocapture %"callee'", double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast double (double)* %callee to i8*
; CHECK-NEXT:   %1 = bitcast double (double)* %"callee'" to i8*
; CHECK-NEXT:   %2 = icmp eq i8* %0, %1
; CHECK-NEXT:   br i1 %2, label %error.i, label %__enzyme_runtimeinactiveerr.exit

; CHECK: error.i:                                          ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @puts(i8* getelementptr inbounds ([79 x i8], [79 x i8]* @.str, i32 0, i32 0))
; CHECK-NEXT:   call void @exit(i32 1)
; CHECK-NEXT:   unreachable

; CHECK: __enzyme_runtimeinactiveerr.exit:                 ; preds = %entry
; CHECK-NEXT:   %[[augloc:.+]] = bitcast double (double)* %"callee'" to { i8*, double } (double)**
; CHECK-NEXT:   %[[augmentptr:.+]] = load { i8*, double } (double)*, { i8*, double } (double)** %[[augloc]]
; CHECK-NEXT:   %call_augmented = call { i8*, double } %[[augmentptr]](double %x)
; CHECK-NEXT:   %[[tape:.+]] = extractvalue { i8*, double } %call_augmented, 0
; CHECK-NEXT:   %[[dcst:.+]] = bitcast double (double)* %"callee'" to { double } (double, double, i8*)**
; CHECK-NEXT:   %[[dptrloc:.+]] = getelementptr { double } (double, double, i8*)*, { double } (double, double, i8*)** %[[dcst]], i64 1
; CHECK-NEXT:   %[[diffeptr:.+]] = load { double } (double, double, i8*)*, { double } (double, double, i8*)** %[[dptrloc]]
; CHECK-NEXT:   %[[ret:.+]] = call { double } %[[diffeptr]](double %x, double %differeturn, i8* %[[tape]])
; CHECK-NEXT:   ret { double } %[[ret:.+]]
; CHECK-NEXT: }
