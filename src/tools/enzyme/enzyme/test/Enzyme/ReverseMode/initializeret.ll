; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -sroa -early-cse -adce -instsimplify -adce -simplifycfg -S -instsimplify | FileCheck %s

; #include <math.h>
;
; __attribute__((noinline))
; double allocateAndSet(double ** arrayp, const double x, unsigned int n) {
;     *arrayp = (double*)malloc(sizeof(double)*n);
;     (*arrayp)[3] = x;
;     return x;
; }
;
; __attribute__((noinline))
; double get(double* x, unsigned int i) {
;     return x[i];
; }
;
; double function(const double x, unsigned int n) {
;     double *array;
;     double as = allocateAndSet(&array, x, n);
;     return get(array, 3) + as;
; }
;
; __attribute__((noinline))
; double derivative(const double x, unsigned int n) {
;     return __builtin_autodiff(function, x, n);
; }
;
; #include <stdio.h>
; #include <stdlib.h>
; int main(int argc, char** argv) {
;     double x = atof(argv[1]);
;     double n = atof(argv[2]);
;     double xp = derivative(x, n);
;     printf("x=%f xp=%f\n", x, xp);
; }
@.str = private unnamed_addr constant [12 x i8] c"x=%f xp=%f\0A\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local double @allocateAndSet(double** nocapture %arrayp, double returned %x, i32 %n) local_unnamed_addr #0 {
entry:
  %conv = zext i32 %n to i64
  %mul = shl nuw nsw i64 %conv, 3
  %call = tail call i8* @malloc(i64 %mul)
  %0 = bitcast double** %arrayp to i8**
  store i8* %call, i8** %0, align 8, !tbaa !2
  %arrayidx = getelementptr inbounds i8, i8* %call, i64 24
  %1 = bitcast i8* %arrayidx to double*
  store double %x, double* %1, align 8, !tbaa !6
  ret double %x
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @get(double* nocapture readonly %x, i32 %i) local_unnamed_addr #2 {
entry:
  %idxprom = zext i32 %i to i64
  %arrayidx = getelementptr inbounds double, double* %x, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8, !tbaa !6
  ret double %0
}

; Function Attrs: nounwind uwtable
define dso_local double @function(double %x, i32 %n) #3 {
entry:
  %array = alloca double*, align 8
  %0 = bitcast double** %array to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #5
  %call = call fast double @allocateAndSet(double** nonnull %array, double %x, i32 %n)
  %1 = load double*, double** %array, align 8, !tbaa !2
  %call1 = tail call fast double @get(double* %1, i32 3)
  %add = fadd fast double %call1, %call
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #5
  ret double %add
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x, i32 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (double (double, i32)*, ...) @__enzyme_autodiff(double (double, i32)* nonnull @function, double %x, i32 %n)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i32)*, ...) #5

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #3 {
entry:
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !2
  %call.i = tail call fast double @strtod(i8* nocapture nonnull %0, i8** null) #5
  %arrayidx1 = getelementptr inbounds i8*, i8** %argv, i64 2
  %1 = load i8*, i8** %arrayidx1, align 8, !tbaa !2
  %call.i10 = tail call fast double @strtod(i8* nocapture nonnull %1, i8** null) #5
  %conv = fptoui double %call.i10 to i32
  %call3 = tail call fast double @derivative(double %call.i, i32 %conv)
  %call4 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i64 0, i64 0), double %call.i, double %call3)
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local double @strtod(i8* readonly, i8** nocapture) local_unnamed_addr #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}


; CHECK: define dso_local {{(dso_local )?}}double @derivative(double %x, i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"array'ipa.i" = alloca double*, align 8
; CHECK-NEXT:   %array.i = alloca double*, align 8
; CHECK-NEXT:   %[[api8:.+]] = bitcast double** %"array'ipa.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8*{{( nonnull)?}} %[[api8]])
; CHECK-NEXT:   %[[ai8:.+]] = bitcast double** %array.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8*{{( nonnull)?}} %[[ai8]])
; CHECK-NEXT:   store double* null, double** %"array'ipa.i", align 8
; CHECK-NEXT:   %[[aug_aas:.+]] = call { i8*, i8* } @augmented_allocateAndSet(double**{{( nonnull)?}} %array.i, double**{{( nonnull)?}} %"array'ipa.i", double %x, i32 %n)
; CHECK-NEXT:   %"'ipl.i" = load double*, double** %"array'ipa.i", align 8
; CHECK-NEXT:   %[[primal:.+]] = load double*, double** %array.i, align 8
; CHECK-NEXT:   call void @diffeget(double* %[[primal]], double* %"'ipl.i", i32 3, double 1.000000e+00)
; CHECK-NEXT:   %[[result:.+]] = call { double } @diffeallocateAndSet(double**{{( nonnull)?}} %array.i, double**{{( nonnull)?}} %"array'ipa.i", double %x, i32 %n, double 1.000000e+00, { i8*, i8* } %[[aug_aas]])
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8*{{( nonnull)?}} %[[api8]])
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8*{{( nonnull)?}} %[[ai8]])
; CHECK-NEXT:   %[[ext:.+]] = extractvalue { double } %[[result]], 0
; CHECK-NEXT:   ret double %[[ext]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @diffeget(double* nocapture readonly %x, double* nocapture %"x'", i32 %i, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[prom:.+]] = zext i32 %i to i64
; CHECK-NEXT:   %[[arrayidx:.+]] = getelementptr inbounds double, double* %"x'", i64 %[[prom]]
; CHECK-NEXT:   %0 = load double, double* %[[arrayidx]], align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %[[arrayidx]], align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ i8*, i8* } @augmented_allocateAndSet(double** nocapture %arrayp, double** nocapture %"arrayp'", double %x, i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %conv = zext i32 %n to i64
; CHECK-NEXT:   %mul = shl nuw nsw i64 %conv, 3
; CHECK-NEXT:   %call = tail call i8* @malloc(i64 %mul)
; CHECK-NEXT:   %[[dcall:.+]] = {{(tail call noalias nonnull i8\* @malloc\(i64 %mul\)( #[0-9]+)?[[:space:]].*call void @llvm.memset.p0i8.i64\(i8\* nonnull (align 1 )?%"call'mi", i8 0, i64 %mul, (i32 1, )?i1 false\)|call i8\* @calloc\(i64 1, i64 %mul\))}}
; CHECK-NEXT:   %"'ipc" = bitcast double** %"arrayp'" to i8**
; CHECK-NEXT:   %0 = bitcast double** %arrayp to i8**
; CHECK-NEXT:   store i8* %[[dcall]], i8** %"'ipc", align 8
; CHECK-NEXT:   store i8* %call, i8** %0, align 8, !tbaa !2
; CHECK-NEXT:   %arrayidx = getelementptr inbounds i8, i8* %call, i64 24
; CHECK-NEXT:   %1 = bitcast i8* %arrayidx to double*
; CHECK-NEXT:   store double %x, double* %1, align 8, !tbaa !6
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i8*, i8* } {{(undef|poison)}}, i8* %[[dcall]], 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i8*, i8* } %.fca.0.insert, i8* %call, 1
; CHECK-NEXT:   ret { i8*, i8* } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double } @diffeallocateAndSet(double** nocapture %arrayp, double** nocapture %"arrayp'", double %x, i32 %n, double %differeturn, { i8*, i8* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = extractvalue { i8*, i8* } %tapeArg, 1
; CHECK-NEXT:   %[[callp:.+]] = extractvalue { i8*, i8* } %tapeArg, 0
; CHECK-NEXT:   %[[arrayidx:.+]] = getelementptr inbounds i8, i8* %[[callp]], i64 24
; CHECK-NEXT:   %[[ipc:.+]] = bitcast i8* %[[arrayidx]] to double*
; CHECK-NEXT:   %[[loaded:.+]] = load double, double* %[[ipc]], align 8
; TODO DSE SHOULD ELIM
; CHECK-NEXT:   store double 0.000000e+00, double* %"'ipc"
; CHECK-NEXT:   %[[result:.+]] = fadd fast double %differeturn, %[[loaded]]
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[callp]])
; CHECK-NEXT:   tail call void @free(i8* %call)
; CHECK-NEXT:   %[[ins:.+]] = insertvalue { double } undef, double %[[result]], 0
; CHECK-NEXT:   ret { double } %[[ins]]
; CHECK-NEXT: }
