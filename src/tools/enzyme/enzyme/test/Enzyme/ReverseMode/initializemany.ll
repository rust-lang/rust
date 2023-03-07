; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -sroa -adce -instcombine -instsimplify -correlated-propagation -early-cse-memssa -instcombine -loop-deletion -simplifycfg -S | FileCheck %s

@.str = private unnamed_addr constant [12 x i8] c"x=%f xp=%f\0A\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local void @allocateAndSet(double** nocapture %arrayp, double %x, i32 %n) local_unnamed_addr #0 {
entry:
  %0 = add i32 %n, 1
  %wide.trip.count = zext i32 %0 to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %call = tail call noalias i8* @malloc(i64 8) #4
  %1 = bitcast i8* %call to double*
  %arrayidx = getelementptr inbounds double*, double** %arrayp, i64 %indvars.iv
  %2 = bitcast double** %arrayidx to i8**
  store i8* %call, i8** %2, align 8, !tbaa !2
  store double %x, double* %1, align 8, !tbaa !6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @get(double** nocapture readonly %x, i32 %i) local_unnamed_addr #2 {
entry:
  %idxprom = zext i32 %i to i64
  %arrayidx = getelementptr inbounds double*, double** %x, i64 %idxprom
  %0 = load double*, double** %arrayidx, align 8, !tbaa !2
  %1 = load double, double* %0, align 8, !tbaa !6
  ret double %1
}

; Function Attrs: nounwind uwtable
define dso_local double @function(double %x, i32 %n) #3 {
entry:
  %add = add i32 %n, 1
  %0 = zext i32 %add to i64
  %vla = alloca double*, i64 %0, align 16
  call void @allocateAndSet(double** nonnull %vla, double %x, i32 %n)
  %call = call fast double @get(double** nonnull %vla, i32 3)
  ret double %call
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x, i32 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (double (double, i32)*, ...) @__enzyme_autodiff(double (double, i32)* nonnull @function, double %x, i32 %n)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i32)*, ...) #4

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #3 {
entry:
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !2
  %call.i = tail call fast double @strtod(i8* nocapture nonnull %0, i8** null) #4
  %arrayidx1 = getelementptr inbounds i8*, i8** %argv, i64 2
  %1 = load i8*, i8** %arrayidx1, align 8, !tbaa !2
  %call.i10 = tail call fast double @strtod(i8* nocapture nonnull %1, i8** null) #4
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
attributes #4 = { nounwind }

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



; CHECK: define internal {{(dso_local )?}}{ double } @diffefunction(double %x, i32 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %add = add i32 %n, 1
; CHECK-NEXT:   %0 = zext i32 %add to i64
; CHECK-NEXT:   %"vla'ipa" = alloca double*, i64 %0, align 16
; CHECK-NEXT:   %1 = bitcast double** %"vla'ipa" to i8*
; CHECK-NEXT:   %2 = shl nuw nsw i64 %0, 3
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull {{(align 16 )?}}%1, i8 0, i64 %2, {{(i32 16, )?}}i1 false)
; CHECK-NEXT:   %vla = alloca double*, i64 %0, align 16
; CHECK-NEXT:   %[[aug_aas:.+]] = call { i8**, i8** } @augmented_allocateAndSet(double** nonnull %vla, double** nonnull %"vla'ipa", double %x, i32 %n)
; CHECK-NEXT:   call void @diffeget(double** nonnull %vla, double** nonnull %"vla'ipa", i32 3, double %differeturn)
; CHECK-NEXT:   %[[ret:.+]] = call { double } @diffeallocateAndSet(double** nonnull %vla, double** nonnull %"vla'ipa", double %x, i32 %n, { i8**, i8** } %[[aug_aas]])
; CHECK-NEXT:   ret { double } %[[ret]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @diffeget(double** nocapture readonly %x, double** nocapture %"x'", i32 %i, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %idxprom = zext i32 %i to i64
; CHECK-NEXT:   %[[arrayidxipge:.+]] = getelementptr inbounds double*, double** %"x'", i64 %idxprom
; CHECK-NEXT:   %"'ipl" = load double*, double** %[[arrayidxipge]], align 8
; CHECK-NEXT:   %0 = load double, double* %"'ipl", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"'ipl", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ i8**, i8** } @augmented_allocateAndSet(double** nocapture %arrayp, double** nocapture %"arrayp'", double %x, i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i32 %n, 1
; CHECK-NEXT:   %wide.trip.count = zext i32 %0 to i64
; CHECK-NEXT:   %mallocsize = shl nuw nsw i64 %wide.trip.count, 3
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %"call'mi_malloccache" = bitcast i8* %malloccall to i8**
; CHECK-NEXT:   %[[malloccall2:.+]] = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %call_malloccache = bitcast i8* %[[malloccall2]] to i8**
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.body
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i8**, i8** } {{(undef|poison)}}, i8** %"call'mi_malloccache", 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i8**, i8** } %.fca.0.insert, i8** %call_malloccache, 1
; CHECK-NEXT:   ret { i8**, i8** } %.fca.1.insert


; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %[[iv:.+]] = phi i64 [ %[[ivnext:.+]], %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %[[ivnext:.+]] = add nuw nsw i64 %[[iv]], 1
; CHECK-NEXT:   %call = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %"call'mi" = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %[[storeloc:.+]] = bitcast i8* %"call'mi" to i64*
; CHECK-NEXT:   store i64 0, i64* %[[storeloc]], align 1
; CHECK-NEXT:   %[[bitcaster:.+]] = bitcast i8* %call to double*
; CHECK-NEXT:   %[[arrayidxipg:.+]] = getelementptr inbounds double*, double** %"arrayp'", i64 %[[iv]]
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double*, double** %arrayp, i64 %[[iv]]
; CHECK-NEXT:   %"'ipc" = bitcast double** %[[arrayidxipg]] to i8**
; CHECK-NEXT:   %[[bctwo:.+]] = bitcast double** %arrayidx to i8**
; CHECK-NEXT:   store i8* %"call'mi", i8** %"'ipc", align 8
; CHECK-NEXT:   %[[gepprimal:.+]] = getelementptr inbounds i8*, i8** %call_malloccache, i64 %iv
; CHECK-NEXT:   store i8* %call, i8** %[[gepprimal]], align 8, !invariant.group !
; CHECK-NEXT:   %[[geper:.+]] = getelementptr inbounds i8*, i8** %"call'mi_malloccache", i64 %[[iv]]
; CHECK-NEXT:   store i8* %"call'mi", i8** %[[geper]], align 8
; CHECK-NEXT:   store i8* %call, i8** %[[bctwo]], align 8, !tbaa !2
; CHECK-NEXT:   store double %x, double* %[[bitcaster]], align 8, !tbaa !6
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq i64 %[[ivnext]], %wide.trip.count
; CHECK-NEXT:   br i1 %[[cmp]], label %for.cond.cleanup, label %for.body
; CHECK-NEXT: }


; CHECK: define internal {{(dso_local )?}}{ double } @diffeallocateAndSet(double** nocapture %arrayp, double** nocapture %"arrayp'", double %x, i32 %n, { i8**, i8** } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { i8**, i8** } %tapeArg, 0
; CHECK-NEXT:   %1 = extractvalue { i8**, i8** } %tapeArg, 1
; CHECK-NEXT:   %[[n1:.+]] = add i32 %n, 1
; CHECK-NEXT:   %wide.trip.count = zext i32 %[[n1]] to i64
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   %[[lcssa:.+]] = phi double [ %[[added:.+]], %invertfor.body ]
; CHECK-NEXT:   %[[toreturn:.+]] = insertvalue { double } undef, double %[[lcssa]], 0
; CHECK-NEXT:   %[[tofree:.+]] = bitcast i8** %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[tofree]])
; CHECK-NEXT:   %[[tofree2:.+]] = bitcast i8** %1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[tofree2]])
; CHECK-NEXT:   ret { double } %[[toreturn]]

; CHECK: invertfor.body:                                   ; preds = %invertfor.body, %entry
; CHECK-NEXT:   %"x'de.0" = phi double [ 0.000000e+00, %entry ], [ %[[added]], %invertfor.body ]
; CHECK-NEXT:   %[[antivar:.+]] = phi i64 [ %wide.trip.count, %entry ], [ %[[sub:.+]], %invertfor.body ]
; CHECK-NEXT:   %[[sub]] = add nsw i64 %[[antivar]], -1
; CHECK-NEXT:   %[[geper:.+]] = getelementptr inbounds i8*, i8** %0, i64 %[[sub]]
; CHECK-NEXT:   %[[metaload:.+]] = load i8*, i8** %[[geper]], align 8
; CHECK-NEXT:   %[[bc:.+]] = bitcast i8* %[[metaload]] to double*
; CHECK-NEXT:   %[[load:.+]] = load double, double* %[[bc]], align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %[[bc]], align 8
; CHECK-NEXT:   %[[added]] = fadd fast double %"x'de.0", %[[load]]
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[metaload]])
; CHECK-NEXT:   %[[_unwrap8:.+]] = getelementptr inbounds i8*, i8** %1, i64 %"iv'ac.0"
; CHECK-NEXT:   %call_unwrap = load i8*, i8** %[[_unwrap8]], align 8, !invariant.group !
; CHECK-NEXT:   tail call void @free(i8* %call_unwrap)
; CHECK-NEXT:   %[[lcmp:.+]] = icmp eq i64 %[[sub]], 0
; CHECK-NEXT:   br i1 %[[lcmp]], label %invertentry, label %invertfor.body
; CHECK-NEXT: }
