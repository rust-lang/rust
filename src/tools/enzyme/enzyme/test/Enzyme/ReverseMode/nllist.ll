; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -early-cse-memssa -instcombine -instsimplify -simplifycfg -adce -licm -correlated-propagation -instcombine -correlated-propagation -adce -instsimplify -correlated-propagation -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s

; #include <stdlib.h>
; #include <stdio.h>
;
; struct n {
;     double *values;
;     struct n *next;
; };
;
; __attribute__((noinline))
; double sum_list(const struct n *__restrict node, unsigned long times) {
;     double sum = 0;
;     for(const struct n *val = node; val != 0; val = val->next) {
;         for(int i=0; i<=times; i++) {
;             sum += val->values[i];
;         }
;     }
;     return sum;
; }
;
; double list_creator(double x, unsigned long n, unsigned long times) {
;     struct n *list = 0;
;     for(int i=0; i<=n; i++) {
;         struct n *newnode = malloc(sizeof(struct n));
;         newnode->next = list;
;         newnode->values = malloc(sizeof(double)*(times+1));
;         for(int j=0; j<=times; j++) {
;             newnode->values[j] = x;
;         }
;         list = newnode;
;     }
;     return sum_list(list, times);
; }
;
; __attribute__((noinline))
; double derivative(double x, unsigned long n, unsigned long times) {
;     return __builtin_autodiff(list_creator, x, n, times);
; }
;
; int main(int argc, char** argv) {
;     double x = atof(argv[1]);
;     unsigned long n = atoi(argv[2]);
;     unsigned long times = atoi(argv[3]);
;     printf("x=%f\n", x);
;     double xp = derivative(x, n, times);
;     printf("xp=%f\n", xp);
;     return 0;
; }

%struct.n = type { double*, %struct.n* }

@.str = private unnamed_addr constant [6 x i8] c"x=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"xp=%f\0A\00", align 1

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @sum_list(%struct.n* noalias readonly %node, i64 %times) local_unnamed_addr #0 {
entry:
  %cmp18 = icmp eq %struct.n* %node, null
  br i1 %cmp18, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup4, %entry
  %val.020 = phi %struct.n* [ %1, %for.cond.cleanup4 ], [ %node, %entry ]
  %sum.019 = phi double [ %add, %for.cond.cleanup4 ], [ 0.000000e+00, %entry ]
  %values = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 0
  %0 = load double*, double** %values, align 8, !tbaa !2
  br label %for.body5

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4, %entry
  %sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.cond.cleanup4 ]
  ret double %sum.0.lcssa

for.cond.cleanup4:                                ; preds = %for.body5
  %next = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 1
  %1 = load %struct.n*, %struct.n** %next, align 8, !tbaa !7
  %cmp = icmp eq %struct.n* %1, null
  br i1 %cmp, label %for.cond.cleanup, label %for.cond1.preheader

for.body5:                                        ; preds = %for.body5, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body5 ]
  %sum.116 = phi double [ %sum.019, %for.cond1.preheader ], [ %add, %for.body5 ]
  %arrayidx = getelementptr inbounds double, double* %0, i64 %indvars.iv
  %2 = load double, double* %arrayidx, align 8, !tbaa !8
  %add = fadd fast double %2, %sum.116
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %times
  br i1 %exitcond, label %for.cond.cleanup4, label %for.body5
}

; Function Attrs: nounwind uwtable
define dso_local double @list_creator(double %x, i64 %n, i64 %times) #1 {
entry:
  %add = shl i64 %times, 3
  %mul = add i64 %add, 8
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup7
  %call13 = tail call fast double @sum_list(%struct.n* %2, i64 %times)
  ret double %call13

for.body:                                         ; preds = %for.cond.cleanup7, %entry
  %indvars.iv30 = phi i64 [ 0, %entry ], [ %indvars.iv.next31, %for.cond.cleanup7 ]
  %list.029 = phi %struct.n* [ null, %entry ], [ %2, %for.cond.cleanup7 ]
  %call = tail call noalias i8* @malloc(i64 16) #4
  %next = getelementptr inbounds i8, i8* %call, i64 8
  %0 = bitcast i8* %next to %struct.n**
  store %struct.n* %list.029, %struct.n** %0, align 8, !tbaa !7
  %call2 = tail call noalias i8* @malloc(i64 %mul) #4
  %1 = bitcast i8* %call to i8**
  store i8* %call2, i8** %1, align 8, !tbaa !2
  %.cast = bitcast i8* %call2 to double*
  br label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8
  %2 = bitcast i8* %call to %struct.n*
  %indvars.iv.next31 = add nuw i64 %indvars.iv30, 1
  %exitcond32 = icmp eq i64 %indvars.iv30, %n
  br i1 %exitcond32, label %for.cond.cleanup, label %for.body

for.body8:                                        ; preds = %for.body8, %for.body
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx = getelementptr inbounds double, double* %.cast, i64 %indvars.iv
  store double %x, double* %arrayidx, align 8, !tbaa !8
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %times
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x, i64 %n, i64 %times) local_unnamed_addr #3 {
entry:
  %0 = tail call double (double (double, i64, i64)*, ...) @__enzyme_autodiff(double (double, i64, i64)* nonnull @list_creator, double %x, i64 %n, i64 %times)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i64, i64)*, ...) #4

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #1 {
entry:
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !10
  %call.i = tail call fast double @strtod(i8* nocapture nonnull %0, i8** null) #4
  %arrayidx1 = getelementptr inbounds i8*, i8** %argv, i64 2
  %1 = load i8*, i8** %arrayidx1, align 8, !tbaa !10
  %call.i16 = tail call i64 @strtol(i8* nocapture nonnull %1, i8** null, i32 10) #4
  %sext = shl i64 %call.i16, 32
  %conv = ashr exact i64 %sext, 32
  %arrayidx3 = getelementptr inbounds i8*, i8** %argv, i64 3
  %2 = load i8*, i8** %arrayidx3, align 8, !tbaa !10
  %call.i17 = tail call i64 @strtol(i8* nocapture nonnull %2, i8** null, i32 10) #4
  %sext19 = shl i64 %call.i17, 32
  %conv5 = ashr exact i64 %sext19, 32
  %call6 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), double %call.i)
  %call7 = tail call fast double @derivative(double %call.i, i64 %conv, i64 %conv5)
  %call8 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i64 0, i64 0), double %call7)
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local double @strtod(i8* readonly, i8** nocapture) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #2

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
!3 = !{!"n", !4, i64 0, !4, i64 8}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !5, i64 0}
!10 = !{!4, !4, i64 0}


; CHECK: define dso_local double @derivative(double %x, i64 %n, i64 %times)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %add.i = shl i64 %times, 3
; CHECK-NEXT:   %mul.i = add i64 %add.i, 8
; CHECK-NEXT:   %0 = shl i64 %n, 3
; CHECK-NEXT:   %mallocsize.i = add i64 %0, 8
; CHECK-NEXT:   %[[mallocforcall2p:.+]] = call noalias nonnull i8* @malloc(i64 %mallocsize.i)
; CHECK-NEXT:   %"call2'mi_malloccache.i" = bitcast i8* %[[mallocforcall2p]] to i8**
; CHECK-NEXT:   %[[mallocforcall2:.+]] = call noalias nonnull i8* @malloc(i64 %mallocsize.i)
; CHECK-NEXT:   %call2_malloccache.i = bitcast i8* %[[mallocforcall2]] to i8**
; CHECK-NEXT:   %[[mallocforcallp:.+]] = call noalias nonnull i8* @malloc(i64 %mallocsize.i)
; CHECK-NEXT:   %"call'mi_malloccache.i" = bitcast i8* %[[mallocforcallp:.+]] to i8**
; CHECK-NEXT:   %[[mcall2:.+]] = call noalias nonnull i8* @malloc(i64 %mallocsize.i)
; CHECK-NEXT:   %call_malloccache.i = bitcast i8* %[[mcall2]] to i8**
; CHECK-NEXT:   br label %for.body.i

; CHECK: [[invertforcondcleanup:.+]]:                         ; preds = %for.cond.cleanup7.i
; CHECK-NEXT:   call void @diffesum_list(%struct.n* nonnull %[[bccast:.+]], %struct.n* nonnull %[[ipci:.+]], i64 %times, double 1.000000e+00)
; CHECK-NEXT:   br label %invertfor.cond.cleanup7.i

; CHECK: for.body.i:                                       ; preds = %for.cond.cleanup7.i, %entry
; CHECK-NEXT:   %[[iv:.+]] = phi i64 [ %[[nextvar:.+]], %for.cond.cleanup7.i ], [ 0, %entry ]
; CHECK-NEXT:   %[[dstruct:.+]] = phi %struct.n* [ %[[ipci]], %for.cond.cleanup7.i ], [ null, %entry ]
; CHECK-NEXT:   %list.029.i = phi %struct.n* [ %[[bccast]], %for.cond.cleanup7.i ], [ null, %entry ]
; CHECK-NEXT:   %[[nextvar]] = add nuw nsw i64 %[[iv]], 1
; CHECK-NEXT:   %call.i = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   %"call'mi.i" = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* {{(noundef )?}}nonnull {{(align 1 )?}}dereferenceable(16) dereferenceable_or_null(16) %"call'mi.i", i8 0, i64 16, {{(i32 1, )?}}i1 false)
; CHECK-NEXT:   %[[nextipgi:.+]] = getelementptr inbounds i8, i8* %"call'mi.i", i64 8
; CHECK-NEXT:   %next.i = getelementptr inbounds i8, i8* %call.i, i64 8
; CHECK-NEXT:   %[[thisipc:.+]] = bitcast i8* %[[nextipgi]] to %struct.n**
; CHECK-NEXT:   %[[bc4:.+]] = bitcast i8* %next.i to %struct.n**
; CHECK-NEXT:   store %struct.n* %[[dstruct]], %struct.n** %[[thisipc]]
; CHECK-NEXT:   %[[callgep:.+]] = getelementptr inbounds i8*, i8** %call_malloccache.i, i64 %[[iv]]
; CHECK-NEXT:   store i8* %call.i, i8** %[[callgep]]
; CHECK-NEXT:   %[[callpgep:.+]] = getelementptr inbounds i8*, i8** %"call'mi_malloccache.i", i64 %[[iv]]
; CHECK-NEXT:   store i8* %"call'mi.i", i8** %[[callpgep]]
; CHECK-NEXT:   store %struct.n* %list.029.i, %struct.n** %[[bc4]], align 8, !tbaa !7
; CHECK-NEXT:   %call2.i = call noalias i8* @malloc(i64 %mul.i)
; CHECK-NEXT:   %"call2'mi.i" = call noalias nonnull i8* @malloc(i64 %mul.i)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull {{(align 1 )?}}%"call2'mi.i", i8 0, i64 %mul.i, {{(i32 1, )?}}i1 false)
; CHECK-NEXT:   %[[thatipc:.+]] = bitcast i8* %"call'mi.i" to i8**
; CHECK-NEXT:   %[[herebc:.+]] = bitcast i8* %call.i to i8**
; CHECK-NEXT:   store i8* %"call2'mi.i", i8** %[[thatipc]], align 8
; CHECK-NEXT:   %[[call2gep:.+]] = getelementptr inbounds i8*, i8** %call2_malloccache.i, i64 %[[iv]]
; CHECK-NEXT:   store i8* %call2.i, i8** %[[call2gep]]
; CHECK-NEXT:   %[[call2pgep:.+]] = getelementptr inbounds i8*, i8** %"call2'mi_malloccache.i", i64 %[[iv]]
; CHECK-NEXT:   store i8* %"call2'mi.i", i8** %[[call2pgep]]
; CHECK-NEXT:   store i8* %call2.i, i8** %[[herebc]], align 8, !tbaa !2
; CHECK-NEXT:   %.cast.i = bitcast i8* %call2.i to double*
; CHECK-NEXT:   br label %for.body8.i

; CHECK: for.cond.cleanup7.i:                              ; preds = %for.body8.i
; CHECK-NEXT:   %[[ipci]] = bitcast i8* %"call'mi.i" to %struct.n*
; CHECK-NEXT:   %[[bccast]] = bitcast i8* %call.i to %struct.n*
; CHECK-NEXT:   %[[hcmp:.+]] = icmp eq i64 %[[iv]], %n
; CHECK-NEXT:   br i1 %[[hcmp]], label %[[invertforcondcleanup]], label %for.body.i

; CHECK: for.body8.i:                                      ; preds = %for.body8.i, %for.body.i
; CHECK-NEXT:   %[[iv2:.+]] = phi i64 [ %[[iv2next:.+]], %for.body8.i ], [ 0, %for.body.i ]
; CHECK-NEXT:   %[[iv2next]] = add nuw nsw i64 %[[iv2]], 1
; CHECK-NEXT:   %arrayidx.i = getelementptr inbounds double, double* %.cast.i, i64 %[[iv2]]
; CHECK-NEXT:   store double %x, double* %arrayidx.i, align 8, !tbaa !8
; CHECK-NEXT:   %[[thiscmp:.+]] = icmp eq i64 %[[iv2]], %times
; CHECK-NEXT:   br i1 %[[thiscmp]], label %for.cond.cleanup7.i, label %for.body8.i

; CHECK: invertfor.body.i:                                 ; preds = %invertfor.body8.i
; CHECK-NEXT:   call void @free(i8* nonnull %[[loadcall2p:.+]])
; CHECK-NEXT:   %[[call2gep:.+]] = getelementptr inbounds i8*, i8** %call2_malloccache.i, i64 %[[antiiv:.+]]
; CHECK-NEXT:   %[[call2ptr:.+]] = load i8*, i8** %[[call2gep]]
; CHECK-NEXT:   call void @free(i8* %[[call2ptr]])
; CHECK-NEXT:   %[[callpgep:.+]] = getelementptr inbounds i8*, i8** %"call'mi_malloccache.i", i64 %[[antiiv]]
; CHECK-NEXT:   %[[callpptr:.+]] = load i8*, i8** %[[callpgep]]
; CHECK-NEXT:   call void @free(i8* nonnull %[[callpptr]])
; CHECK-NEXT:   %[[callgep:.+]] = getelementptr inbounds i8*, i8** %call_malloccache.i, i64 %[[antiiv]]
; CHECK-NEXT:   %[[callptr:.+]] = load i8*, i8** %[[callgep]]
; CHECK-NEXT:   call void @free(i8* %[[callptr]])
; CHECK-NEXT:   %[[cmpne:.+]] = icmp eq i64 %[[antiiv]], 0
; CHECK-NEXT:   br i1 %[[cmpne]], label %diffelist_creator.exit, label %incinvertfor.body.i

; CHECK: incinvertfor.body.i:
; CHECK-NEXT:   %[[iv7sub:.+]] = add nsw i64 %[[antiiv]], -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup7.i

; CHECK: invertfor.cond.cleanup7.i:
; CHECK-NEXT:   %"x'de.0.i" = phi double [ 0.000000e+00, %[[invertforcondcleanup]] ], [ %[[faddloop:.+]], %incinvertfor.body.i ]
; CHECK-NEXT:   %[[antiiv]] = phi i64 [ %n, %[[invertforcondcleanup]] ], [ %[[iv7sub]], %incinvertfor.body.i ]
; CHECK-NEXT:   %[[midcall2pgep:.+]] = getelementptr inbounds i8*, i8** %"call2'mi_malloccache.i", i64 %[[antiiv]]
; CHECK-NEXT:   %[[loadcall2p]] = load i8*, i8** %[[midcall2pgep]]
; CHECK-NEXT:   %[[precast:.+]] = bitcast i8* %[[loadcall2p]] to double*
; CHECK-NEXT:   br label %invertfor.body8.i

; CHECK: invertfor.body8.i:
; CHECK-NEXT:   %"x'de.1.i" = phi double [ %"x'de.0.i", %invertfor.cond.cleanup7.i ], [ %[[faddloop]], %incinvertfor.body8.i ]
; CHECK-NEXT:   %[[antiiv2:.+]] = phi i64 [ %times, %invertfor.cond.cleanup7.i ], [ %[[idxsub:.+]], %incinvertfor.body8.i ]
; CHECK-NEXT:   %[[arrayidxipgi:.+]] = getelementptr inbounds double, double* %[[precast]], i64 %[[antiiv2]]
; CHECK-NEXT:   %[[looparray:.+]] = load double, double* %[[arrayidxipgi]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[arrayidxipgi]]
; CHECK-NEXT:   %[[faddloop]] = fadd fast double %"x'de.1.i", %[[looparray]]
; CHECK-NEXT:   %[[loopcmpne:.+]] = icmp eq i64 %[[antiiv2]], 0
; CHECK-NEXT:   br i1 %[[loopcmpne]], label %invertfor.body.i, label %incinvertfor.body8.i

; CHECK: incinvertfor.body8.i:
; CHECK-NEXT:   %[[idxsub]] = add nsw i64 %[[antiiv2]], -1
; CHECK-NEXT:   br label %invertfor.body8.i

; CHECK: diffelist_creator.exit:                           ; preds = %invertfor.body.i
; CHECK-NEXT:   call void @free(i8* nonnull %[[mallocforcall2p]])
; CHECK-NEXT:   call void @free(i8* nonnull %[[mallocforcall2]])
; CHECK-NEXT:   call void @free(i8* nonnull %[[mallocforcallp]])
; CHECK-NEXT:   call void @free(i8* nonnull %[[mcall2]])
; CHECK-NEXT:   ret double %[[faddloop]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @diffesum_list(%struct.n* noalias readonly %node, %struct.n* %"node'", i64 %times, double %differeturn)
; checked separately in sumnllist
