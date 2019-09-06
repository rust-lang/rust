; RUN: opt < %s %loadEnzyme -enzyme -inline -mem2reg -adce -aggressive-instcombine -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -S | FileCheck %s

%struct.n = type { double, %struct.n* }

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @sum_list(%struct.n* noalias readonly %node) #0 {
entry:
  %cmp6 = icmp eq %struct.n* %node, null
  br i1 %cmp6, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  ret double %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %val.08 = phi %struct.n* [ %1, %for.body ], [ %node, %entry ]
  %sum.07 = phi double [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %value = getelementptr inbounds %struct.n, %struct.n* %val.08, i64 0, i32 0
  %0 = load double, double* %value, align 8, !tbaa !2
  %add = fadd fast double %0, %sum.07
  %next = getelementptr inbounds %struct.n, %struct.n* %val.08, i64 0, i32 1
  %1 = load %struct.n*, %struct.n** %next, align 8, !tbaa !8
  %cmp = icmp eq %struct.n* %1, null
  br i1 %cmp, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local double @list_creator(double %x, i64 %n) #1 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %0 = bitcast i8* %call to %struct.n*
  %call2 = tail call fast double @sum_list(%struct.n* %0)
  ret double %call2

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %list.011 = phi %struct.n* [ null, %entry ], [ %1, %for.body ]
  %call = tail call noalias i8* @malloc(i64 16) #4
  %1 = bitcast i8* %call to %struct.n*
  %next = getelementptr inbounds i8, i8* %call, i64 8
  %2 = bitcast i8* %next to %struct.n**
  store %struct.n* %list.011, %struct.n** %2, align 8, !tbaa !8
  %value = bitcast i8* %call to double*
  store double %x, double* %value, align 8, !tbaa !2
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x, i64 %n) local_unnamed_addr #3 {
entry:
  %0 = tail call double (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @list_creator, double %x, i64 %n)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i64)*, ...) #4

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"n", !4, i64 0, !7, i64 8}
!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!"any pointer", !5, i64 0}
!8 = !{!3, !7, i64 8}
!9 = !{!7, !7, i64 0}

; CHECK: define dso_local double @derivative(double %x, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = add nuw i64 %n, 1
; CHECK-NEXT:  %mallocsize.i = mul i64 %0, 8
; CHECK-NEXT:  %[[mallocp:.+]] = call noalias nonnull i8* @malloc(i64 %mallocsize.i) #4
; CHECK-NEXT:  %[[callpcache:.+]] = bitcast i8* %[[mallocp]] to i8**
; CHECK-NEXT:  %[[malloc1:.+]] = call noalias nonnull i8* @malloc(i64 %mallocsize.i) #4
; CHECK-NEXT:  %call_malloccache.i = bitcast i8* %[[malloc1:.+]] to i8**
; CHECK-NEXT:  br label %for.body.i

; CHECK:for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
; CHECK-NEXT:  %1 = phi %struct.n* [ null, %entry ], [ %[[dstructncast:.+]], %for.body.i ]
; CHECK-NEXT:  %list.011.i = phi %struct.n* [ null, %entry ], [ %5, %for.body.i ]
; CHECK-NEXT:  %2 = icmp ult i64 %indvars.iv.i, %n
; CHECK-NEXT:  %call.i = call noalias i8* @malloc(i64 16) #4
; CHECK-NEXT:  %[[callcachegep:.+]] = getelementptr i8*, i8** %call_malloccache.i, i64 %indvars.iv.i
; CHECK-NEXT:  store i8* %call.i, i8** %[[callcachegep]]
; CHECK-NEXT:  %"call'mi.i" = call noalias i8* @malloc(i64 16) #4
; CHECK-NEXT:  call void @llvm.memset.p0i8.i64(i8* nonnull %"call'mi.i", i8 0, i64 16, i1 false) #4
; CHECK-NEXT:  %[[callpcachegep:.+]] = getelementptr i8*, i8** %[[callpcache]], i64 %indvars.iv.i
; CHECK-NEXT:  store i8* %"call'mi.i", i8** %[[callpcachegep]]
; CHECK-NEXT:  %5 = bitcast i8* %call.i to %struct.n*
; CHECK-NEXT:  %next.i = getelementptr inbounds i8, i8* %call.i, i64 8
; CHECK-NEXT:  %6 = bitcast i8* %next.i to %struct.n**
; CHECK-NEXT:  %"next'ipg.i" = getelementptr i8, i8* %"call'mi.i", i64 8
; CHECK-NEXT:  %[[dstruct1:.+]] = bitcast i8* %"next'ipg.i" to %struct.n**
; CHECK-NEXT:  store %struct.n* %1, %struct.n** %[[dstruct1]]
; CHECK-NEXT:  store %struct.n* %list.011.i, %struct.n** %6, align 8, !tbaa !8
; CHECK-NEXT:  %value.i = bitcast i8* %call.i to double*
; CHECK-NEXT:  store double %x, double* %value.i, align 8, !tbaa !2
; CHECK-NEXT:  %indvars.iv.next.i = add nuw i64 %indvars.iv.i, 1
; CHECK-NEXT:  %[[dstructncast]] = bitcast i8* %"call'mi.i" to %struct.n*
; CHECK-NEXT:  br i1 %2, label %for.body.i, label %invertfor.cond.cleanup.i

; CHECK:invertfor.cond.cleanup.i:                         ; preds = %for.body.i
; CHECK-NEXT:  %[[foo:.+]] = call {} @diffesum_list(%struct.n* %5, %struct.n* %[[dstructncast]], double 1.000000e+00) #4
; CHECK-NEXT:  br label %invertfor.body.i

; CHECK:invertfor.body.i:                                 ; preds = %invertfor.body.i, %invertfor.cond.cleanup.i
; CHECK-NEXT:  %"x'de.0.i" = phi double [ 0.000000e+00, %invertfor.cond.cleanup.i ], [ %[[add:.+]], %invertfor.body.i ]
; CHECK-NEXT:  %"indvars.iv'phi.i" = phi i64 [ %n, %invertfor.cond.cleanup.i ], [ %[[sub:.+]], %invertfor.body.i ]
; CHECK-NEXT:  %[[sub]] = sub i64 %"indvars.iv'phi.i", 1
; CHECK-NEXT:  %[[gep:.+]] = getelementptr i8*, i8** %"call'mi_malloccache.i", i64 %"indvars.iv'phi.i"
; CHECK-NEXT:  %[[loadcache:.+]] = load i8*, i8** %[[gep]]
; CHECK-NEXT:  %"value'ipc.i" = bitcast i8* %[[loadcache]] to double*
; CHECK-NEXT:  %[[load:.+]] = load double, double* %"value'ipc.i"
; this store is optional and could get removed by DCE
; CHECK-NEXT:  store double 0.000000e+00, double* %"value'ipc.i"
; CHECK-NEXT:  %[[add]] = fadd fast double %"x'de.0.i", %[[load]]
; CHECK-NEXT:  %[[prefree2:.+]] = load i8*, i8** %[[gep]]
; CHECK-NEXT:  call void @free(i8* nonnull %[[prefree2]]) #4
; CHECK-NEXT:  %[[gepcall:.+]] = getelementptr i8*, i8** %call_malloccache.i, i64 %"indvars.iv'phi.i"
; CHECK-NEXT:  %[[loadprefree:.+]] = load i8*, i8** %[[gepcall]]
; CHECK-NEXT:  call void @free(i8* %[[loadprefree]]) #4
; CHECK-NEXT:  %[[cmp:.+]] = icmp ne i64 %"indvars.iv'phi.i", 0
; CHECK-NEXT:  br i1 %[[cmp:.+]], label %invertfor.body.i, label %diffelist_creator.exit

; CHECK:diffelist_creator.exit:                           ; preds = %invertfor.body.i
; CHECK-NEXT:  call void @free(i8* nonnull %[[malloc1]]) #4
; CHECK-NEXT:  call void @free(i8* nonnull %[[mallocp]]) #4
; CHECK-NEXT:  ret double %[[add]]


; CHECK: define internal {} @diffesum_list(%struct.n* noalias readonly %node, %struct.n* %"node'", double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp6 = icmp eq %struct.n* %node, null
; CHECK-NEXT:   br i1 %cmp6, label %invertfor.cond.cleanup, label %for.body

; CHECK: for.body.preheader:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 8)
; CHECK-NEXT:   %[[malloccache:.+]] = bitcast i8* %malloccall to %struct.n**
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:
; CHECK-NEXT:   %_mdyncache.1 = phi %struct.n** [ %[[malloccache]], %for.body.preheader ], [ %5, %for.body ]
; CHECK-NEXT:   %0 = phi i64 [ %2, %for.body ], [ 0, %for.body.preheader ]
; CHECK-NEXT:   %1 = phi %struct.n* [ %"'ipl", %for.body ], [ %"node'", %for.body.preheader ]
; CHECK-NEXT:   %val.08 = phi %struct.n* [ %7, %for.body ], [ %node, %for.body.preheader ]
; CHECK-NEXT:   %2 = add nuw i64 %0, 1
; CHECK-NEXT:   %3 = bitcast %struct.n** %_mdyncache.1 to i8*
; CHECK-NEXT:   %4 = mul nuw i64 8, %2
; CHECK-NEXT:   %_realloccache = call i8* @realloc(i8* %3, i64 %4)
; CHECK-NEXT:   %5 = bitcast i8* %_realloccache to %struct.n**
; CHECK-NEXT:   %6 = getelementptr %struct.n*, %struct.n** %5, i64 %0
; CHECK-NEXT:   store %struct.n* %1, %struct.n** %6
; CHECK-NEXT:   %next = getelementptr inbounds %struct.n, %struct.n* %val.08, i64 0, i32 1
; CHECK-NEXT:   %"next'ipg" = getelementptr %struct.n, %struct.n* %1, i64 0, i32 1
; CHECK-NEXT:   %"'ipl" = load %struct.n*, %struct.n** %"next'ipg", align 8
; CHECK-NEXT:   %7 = load %struct.n*, %struct.n** %next, align 8, !tbaa !8
; CHECK-NEXT:   %cmp = icmp eq %struct.n* %7, null
; CHECK-NEXT:   br i1 %cmp, label %invertfor.cond.cleanup, label %for.body

; CHECK: invertentry:                                      ; preds = %invertfor.cond.cleanup, %invertfor.body.preheader
; CHECK-NEXT:   ret {} undef

; CHECK: invertfor.body.preheader:                         ; preds = %invertfor.body
; CHECK-NEXT:   %8 = bitcast %struct.n** %_mdyncache.0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %8)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertfor.cond.cleanup: 
; CHECK-NEXT:   %_mdyncache.0 = phi %struct.n** [ undef, %entry ], [ %5, %for.body ]
; CHECK-NEXT:   %_cache.0 = phi i64 [ undef, %entry ], [ %0, %for.body ]
; CHECK-NEXT:   br i1 %cmp6, label %invertentry, label %invertfor.body

; CHECK: invertfor.body:                                   ; preds = %invertfor.cond.cleanup, %invertfor.body
; CHECK-NEXT:   %"'phi" = phi i64 [ %[[subidx:.+]], %invertfor.body ], [ %_cache.0, %invertfor.cond.cleanup ]
; CHECK-NEXT:   %[[subidx]] = sub i64 %"'phi", 1
; CHECK-NEXT:   %[[structptr:.+]] = getelementptr %struct.n*, %struct.n** %_mdyncache.0, i64 %"'phi"
; CHECK-NEXT:   %[[struct:.+]] = load %struct.n*, %struct.n** %[[structptr]]
; CHECK-NEXT:   %"value'ipg" = getelementptr %struct.n, %struct.n* %[[struct]], i64 0, i32 0
; CHECK-NEXT:   %[[val0:.+]] = load double, double* %"value'ipg"
; CHECK-NEXT:   %[[addval:.+]] = fadd fast double %[[val0]], %[[differet]]
; CHECK-NEXT:   store double %[[addval]], double* %"value'ipg"
; CHECK-NEXT:   %[[cmpne:.+]] = icmp ne i64 %"'phi", 0
; CHECK-NEXT:   br i1 %[[cmpne]], label %invertfor.body, label %invertfor.body.preheader
; CHECK-NEXT: }
