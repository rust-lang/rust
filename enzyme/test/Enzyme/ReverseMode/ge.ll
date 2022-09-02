; RUN: if [ %llvmver -lt 14 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -adce -correlated-propagation -simplifycfg -early-cse -S | FileCheck %s -check-prefixes STORE,SHARED; fi
; RUN: if [ %llvmver -ge 14 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -adce -correlated-propagation -simplifycfg -early-cse -S | FileCheck %s -check-prefixes MEMCPY,SHARED; fi

; void __enzyme_autodiff(void*, ...);

; double cache(double* x, unsigned N) {
;     double sum = 0.0;
;     for(unsigned i=0; i<=N; i++) {
;         sum += x[i] * x[i];
;     }
;     x[0] = 0.0;
;     return sum;
; }

; void ad(double* in, double* din, unsigned N) {
;     __enzyme_autodiff(cache, in, din, N);
; }

; ModuleID = 'foo.c'
source_filename = "foo.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define dso_local double @cache(double* nocapture %x, i32 %N) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  store double 0.000000e+00, double* %x, align 8, !tbaa !2
  ret double %add

for.body:                                         ; preds = %entry, %for.body
  %i.013 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum.012 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %idxprom = zext i32 %i.013 to i64
  %arrayidx = getelementptr inbounds double, double* %x, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8, !tbaa !2
  %mul = fmul double %0, %0
  %add = fadd double %sum.012, %mul
  %inc = add i32 %i.013, 1
  %cmp = icmp ugt i32 %inc, %N
  br i1 %cmp, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local void @ad(double* %in, double* %din, i32 %N) local_unnamed_addr #1 {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, i32)* @cache to i8*), double* %in, double* %din, i32 %N) #3
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #2

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 (trunk 336729)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}


; SHARED: define internal void @diffecache(double* nocapture %x, double* nocapture %"x'", i32 %N, double %differeturn)
; SHARED-NEXT: entry:
; TODO-NEXT:   %[[a1:.+]] = zext i32 %N to i64
; SHARED:   %[[a2:.+]] = add{{( nuw)?}}{{( nsw)?}} i64 %[[a1:.+]], 1
; SHARED-NEXT:   %mallocsize = mul nuw nsw i64 %[[a2]], 8
; SHARED-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; SHARED-NEXT:   %_malloccache = bitcast i8* %malloccall to double*
; MEMCPY-NEXT:   %2 = bitcast double* %x to i8*
; MEMCPY-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %malloccall, i8* nonnull align 8 %2, i64 %mallocsize, i1 false)
; SHARED-NEXT:   br label %for.body

; SHARED: for.cond.cleanup:                                 ; preds = %for.body
; SHARED-NEXT:   store double 0.000000e+00, double* %x, align 8, !tbaa !2
; SHARED-NEXT:   store double 0.000000e+00, double* %"x'", align 8
; SHARED-NEXT:   br label %invertfor.body

; SHARED: for.body:                                         ; preds = %for.body, %entry
; SHARED-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; SHARED-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; SHARED-NEXT:   %[[a3:.+]] = trunc i64 %iv to i32
; STORE-NEXT:   %idxprom = zext i32 %[[a3]] to i64
; STORE-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 %idxprom
; STORE-NEXT:   %[[a4:.+]] = load double, double* %arrayidx, align 8, !tbaa !2
; STORE-NEXT:   %[[a5:.+]] = getelementptr inbounds double, double* %_malloccache, i64 %iv
; STORE-NEXT:   store double %[[a4]], double* %[[a5]], align 8, !tbaa !2, !invariant.group !
; SHARED-NEXT:   %inc = add i32 %[[a3]], 1
; SHARED-NEXT:   %cmp = icmp ugt i32 %inc, %N
; SHARED-NEXT:   br i1 %cmp, label %for.cond.cleanup, label %for.body

; SHARED: invertentry:                                      ; preds = %invertfor.body
; SHARED-NEXT:   tail call void @free(i8* nonnull %malloccall)
; SHARED-NEXT:   ret void

; SHARED: invertfor.body:                                   ; preds = %incinvertfor.body, %for.cond.cleanup
; SHARED:   %"add'de.0" = phi double [ %differeturn, %for.cond.cleanup ], [ %"add'de.0", %incinvertfor.body ]
; SHARED:   %"iv'ac.0" = phi i64 [ %[[a1]], %for.cond.cleanup ], [ %[[a12:.+]], %incinvertfor.body ]
; SHARED-NEXT:   %[[a6:.+]] = getelementptr inbounds double, double* %_malloccache, i64 %"iv'ac.0"
; SHARED-NEXT:   %[[a7:.+]] = load double, double* %[[a6]], align 8, {{(!tbaa !2, )?}}!invariant.group !
; SHARED-NEXT:   %m0diffe = fmul fast double %"add'de.0", %[[a7]]
; SHARED-NEXT:   %[[a8:.+]] = fadd fast double %m0diffe, %m0diffe
; SHARED-NEXT:   %[[unwrap:.+]] = trunc i64 %"iv'ac.0" to i32
; SHARED-NEXT:   %idxprom_unwrap = zext i32 %[[unwrap]] to i64
; SHARED-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %idxprom_unwrap
; SHARED-NEXT:   %[[a9:.+]] = load double, double* %"arrayidx'ipg_unwrap", align 8
; SHARED-NEXT:   %[[a10:.+]] = fadd fast double %[[a9]], %[[a8]]
; SHARED-NEXT:   store double %[[a10]], double* %"arrayidx'ipg_unwrap", align 8
; SHARED-NEXT:   %[[a11:.+]] = icmp eq i64 %"iv'ac.0", 0
; SHARED-NEXT:   br i1 %[[a11]], label %invertentry, label %incinvertfor.body

; SHARED: incinvertfor.body:                                ; preds = %invertfor.body
; SHARED-NEXT:   %[[a12]] = add nsw i64 %"iv'ac.0", -1
; SHARED-NEXT:   br label %invertfor.body
; SHARED-NEXT: }
