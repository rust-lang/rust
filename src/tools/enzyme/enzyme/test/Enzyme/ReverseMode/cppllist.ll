; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -adce -instcombine -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -loop-simplify -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s

; #include <stdlib.h>
; #include <stdio.h>
;
; class node {
; public:
;     double value;
;     node *next;
;     node(node* next_, double value_) {
;         value = value_;
;         next = next_;
;     }
; };
;
; __attribute__((noinline))
; double sum_list(const node *__restrict node) {
;     double sum = 0;
;     const class node *val;
;     for(val = node; val != 0; val = val->next) {
;         sum += val->value;
;     }
;     return sum;
; }
;
; double list_creator(double x, unsigned long n) {
;     node *list = 0;
;     for(int i=0; i<=n; i++) {
;         list = new node(list, x);
;     }
;     auto res = sum_list(list);
;     delete list;
;     return res;
; }
;
; __attribute__((noinline))
; double derivative(double x, unsigned long n) {
;     return __builtin_autodiff(list_creator, x, n);
; }
;
; int main(int argc, char** argv) {
;     double x = atof(argv[1]);
;     double n = atoi(argv[2]);
;     printf("x=%f\n", x);
;     double xp = derivative(x, n);
;     printf("xp=%f\n", xp);
;     return 0;
; }

%class.node = type { double, %class.node* }

@.str = private unnamed_addr constant [6 x i8] c"x=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"xp=%f\0A\00", align 1

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @_Z8sum_listPK4node(%class.node* noalias readonly %node) local_unnamed_addr #0 {
entry:
  %cmp6 = icmp eq %class.node* %node, null
  br i1 %cmp6, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %sum.07 = phi double [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %val.08 = phi %class.node* [ %1, %for.body ], [ %node, %entry ]
  %value = getelementptr inbounds %class.node, %class.node* %val.08, i64 0, i32 0
  %0 = load double, double* %value, align 8, !tbaa !2
  %add = fadd fast double %0, %sum.07
  %next = getelementptr inbounds %class.node, %class.node* %val.08, i64 0, i32 1
  %1 = load %class.node*, %class.node** %next, align 8, !tbaa !8
  %cmp = icmp eq %class.node* %1, null
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  ret double %sum.0.lcssa
}

; Function Attrs: nounwind uwtable
define dso_local double @_Z12list_creatordm(double %x, i64 %n) #1 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %list.09 = phi %class.node* [ null, %entry ], [ %0, %for.body ]
  %call = tail call i8* @_Znwm(i64 16) #8
  %0 = bitcast i8* %call to %class.node*
  %value.i = bitcast i8* %call to double*
  store double %x, double* %value.i, align 8, !tbaa !2
  %next.i = getelementptr inbounds i8, i8* %call, i64 8
  %1 = bitcast i8* %next.i to %class.node**
  store %class.node* %list.09, %class.node** %1, align 8, !tbaa !8
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %delete.end, label %for.body

delete.end:                                       ; preds = %for.body
  %2 = bitcast i8* %call to %class.node*
  %call1 = tail call fast double @_Z8sum_listPK4node(%class.node* nonnull %2)
  tail call void @_ZdlPv(i8* nonnull %call) #8
  ret double %call1
}

; Function Attrs: nobuiltin
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #2

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #3

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z10derivativedm(double %x, i64 %n) local_unnamed_addr #4 {
entry:
  %0 = tail call double (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @_Z12list_creatordm, double %x, i64 %n)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i64)*, ...) #5

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #6 {
entry:
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !9
  %call.i = tail call fast double @strtod(i8* nocapture nonnull %0, i8** null) #5
  %arrayidx1 = getelementptr inbounds i8*, i8** %argv, i64 2
  %1 = load i8*, i8** %arrayidx1, align 8, !tbaa !9
  %call.i12 = tail call i64 @strtol(i8* nocapture nonnull %1, i8** null, i32 10) #5
  %call3 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), double %call.i)
  %conv4 = and i64 %call.i12, 4294967295
  %call5 = tail call fast double @_Z10derivativedm(double %call.i, i64 %conv4)
  %call6 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i64 0, i64 0), double %call5)
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #7

; Function Attrs: nounwind
declare dso_local double @strtod(i8* readonly, i8** nocapture) local_unnamed_addr #7

; Function Attrs: nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #7

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind }
attributes #6 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { builtin nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTS4node", !4, i64 0, !7, i64 8}
!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"any pointer", !5, i64 0}
!8 = !{!3, !7, i64 8}
!9 = !{!7, !7, i64 0}


; CHECK: define dso_local double @_Z10derivativedm(double %x, i64 %n) local_unnamed_addr #4 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = shl i64 %n, 3
; CHECK-NEXT:   %mallocsize.i = add i64 %0, 8
; CHECK-NEXT:   %malloccall.i = call noalias nonnull i8* @malloc(i64 %mallocsize.i) #5
; CHECK-NEXT:   %"call'mi_malloccache.i" = bitcast i8* %malloccall.i to i8**
; CHECK-NEXT:   %[[call_malloc:.+]] = call noalias nonnull i8* @malloc(i64 %mallocsize.i) #5
; CHECK-NEXT:   %call_malloccache.i = bitcast i8* %[[call_malloc]] to i8**
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %[[iv:.+]] = phi i64 [ %[[ivnext:.+]], %for.body.i ], [ 0, %entry ]
; CHECK-NEXT:   %[[nodevar:.+]] = phi %class.node* [ %"'ipc.i", %for.body.i ], [ null, %entry ]
; CHECK-NEXT:   %list.09.i = phi %class.node* [ %[[bcnode:.+]], %for.body.i ], [ null, %entry ]
; CHECK-NEXT:   %[[ivnext]] = add nuw nsw i64 %[[iv]], 1
; CHECK-NEXT:   %call.i = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @_Znwm(i64 16) 
; CHECK-NEXT:   %"call'mi.i" = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @_Znwm(i64 16) 
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* {{(noundef )?}}nonnull {{(align 1 )?}}dereferenceable(16) dereferenceable_or_null(16) %"call'mi.i", i8 0, i64 16, {{(i32 1, )?}}i1 false)
; CHECK-NEXT:   %"'ipc.i" = bitcast i8* %"call'mi.i" to %class.node*
; CHECK-NEXT:   %[[bcnode]] = bitcast i8* %call.i to %class.node*
; CHECK-NEXT:   %value.i.i = bitcast i8* %call.i to double*
; CHECK-NEXT:   store double %x, double* %value.i.i, align 8, !tbaa !2
; CHECK-NEXT:   %[[callgep:.+]] = getelementptr inbounds i8*, i8** %call_malloccache.i, i64 %[[iv]]
; CHECK-NEXT:   store i8* %call.i, i8** %[[callgep]]
; CHECK-NEXT:   %[[callpgep:.+]] = getelementptr inbounds i8*, i8** %"call'mi_malloccache.i", i64 %[[iv]]
; CHECK-NEXT:   store i8* %"call'mi.i", i8** %[[callpgep]]
; CHECK-NEXT:   %[[nextiipgi:.+]] = getelementptr inbounds i8, i8* %"call'mi.i", i64 8
; CHECK-NEXT:   %next.i.i = getelementptr inbounds i8, i8* %call.i, i64 8
; CHECK-NEXT:   %"'ipc1.i" = bitcast i8* %[[nextiipgi]] to %class.node**
; CHECK-NEXT:   %[[bctwo:.+]] = bitcast i8* %next.i.i to %class.node**
; CHECK-NEXT:   store %class.node* %[[nodevar]], %class.node** %"'ipc1.i"
; CHECK-NEXT:   store %class.node* %list.09.i, %class.node** %[[bctwo]], align 8, !tbaa !8
; CHECK-NEXT:   %[[endcomp:.+]] = icmp eq i64 %[[iv]], %n
; CHECK-NEXT:   br i1 %[[endcomp]], label %[[invertdelete:.+]], label %for.body.i


; CHECK: [[invertdelete]]:                               ; preds = %for.body.i
; CHECK-NEXT:   call void @diffe_Z8sum_listPK4node(%class.node* nonnull %[[bcnode]], %class.node* nonnull %"'ipc.i", double 1.000000e+00)
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: invertfor.body.i:
; CHECK-NEXT:   %"x'de.0.i" = phi double [ 0.000000e+00, %[[invertdelete:.+]] ], [ %[[xadd:.+]], %incinvertfor.body.i ]
; CHECK-NEXT:   %[[antivar:.+]] = phi i64 [ %n, %[[invertdelete]] ], [ %[[isub:.+]], %incinvertfor.body.i ]
; CHECK-NEXT:   %[[gepiv:.+]] = getelementptr inbounds i8*, i8** %"call'mi_malloccache.i", i64 %[[antivar]]
; CHECK-NEXT:   %[[metaload:.+]] = load i8*, i8** %[[gepiv]]
; CHECK-NEXT:   %[[bcast:.+]] = bitcast i8* %[[metaload]] to double*
; CHECK-NEXT:   %[[load:.+]] = load double, double* %[[bcast]]
; this store is optional and could get removed by DCE
; CHECK-NEXT:   store double 0.000000e+00, double* %[[bcast]]
; CHECK-NEXT:   %[[xadd]] = fadd fast double %"x'de.0.i", %[[load]]
; this reload really should be eliminated
; CHECK-NEXT:   call void @_ZdlPv(i8* nonnull %[[metaload]])
; CHECK-NEXT:   %[[heregep:.+]] = getelementptr inbounds i8*, i8** %call_malloccache.i, i64 %[[antivar]]
; CHECK-NEXT:   %[[callload2free:.+]] = load i8*, i8** %[[heregep]]
; CHECK-NEXT:   call void @_ZdlPv(i8* %[[callload2free]])
; CHECK-NEXT:   %[[cmpinst:.+]] = icmp eq i64 %[[antivar]], 0
; CHECK-NEXT:   br i1 %[[cmpinst]], label %diffe_Z12list_creatordm.exit, label %incinvertfor.body.i

; CHECK: incinvertfor.body.i:
; CHECK-NEXT:   %[[isub]] = add nsw i64 %[[antivar]], -1
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: diffe_Z12list_creatordm.exit:                     ; preds = %invertfor.body.i
; CHECK-NEXT:   call void @free(i8* nonnull %malloccall.i)
; CHECK-NEXT:   call void @free(i8* nonnull %[[call_malloc]])
; CHECK-NEXT:   ret double %[[xadd]]
; CHECK-NEXT: }


; CHECK: define internal {{(dso_local )?}}void @diffe_Z8sum_listPK4node(%class.node* noalias readonly %node, %class.node* %"node'", double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq %class.node* %node, null
; CHECK-NEXT:   br i1 %[[cmp]], label %invertentry, label %for.body

; CHECK: for.body:
; CHECK-NEXT:   %[[rawcache:.+]] = phi i8* [ %[[realloccache:.+]], %[[mergeblk:.+]] ], [ null, %entry ]
; CHECK-NEXT:   %[[preidx:.+]] = phi i64 [ %[[postidx:.+]], %[[mergeblk:.+]] ], [ 0, %entry ]
; CHECK-NEXT:   %[[cur:.+]] = phi %class.node* [ %"'ipl", %[[mergeblk:.+]] ], [ %"node'", %entry ]
; CHECK-NEXT:   %val.08 = phi %class.node* [ %[[loadst:.+]], %[[mergeblk:.+]] ], [ %node, %entry ]
; CHECK-NEXT:   %[[postidx]] = add nuw nsw i64 %[[preidx]], 1

; CHECK-NEXT:   %[[nexttrunc0:.+]] = and i64 %[[postidx]], 1
; CHECK-NEXT:   %[[nexttrunc:.+]] = icmp ne i64 %[[nexttrunc0]], 0
; CHECK-NEXT:   %[[popcnt:.+]] = call i64 @llvm.ctpop.i64(i64 %iv.next)
; CHECK-NEXT:   %[[le2:.+]] = icmp ult i64 %[[popcnt:.+]], 3
; CHECK-NEXT:   %[[shouldgrow:.+]] = and i1 %[[le2]], %[[nexttrunc]]
; CHECK-NEXT:   br i1 %[[shouldgrow]], label %grow.i, label %[[mergeblk]]

; CHECK: grow.i:
; CHECK-NEXT:   %[[ctlz:.+]] = call i64 @llvm.ctlz.i64(i64 %[[postidx]], i1 true)
; CHECK-NEXT:   %[[maxbit:.+]] = sub nuw nsw i64 64, %[[ctlz]]
; CHECK-NEXT:   %[[numbytes:.+]] = shl i64 8, %[[maxbit]]
; CHECK-NEXT:   %[[growalloc:.+]] = call i8* @realloc(i8* %[[rawcache]], i64 %[[numbytes]])
; CHECK-NEXT:   br label %[[mergeblk]]

; CHECK: [[mergeblk]]:
; CHECK-NEXT:   %[[realloccache]] = phi i8* [ %[[growalloc]], %grow.i ], [ %[[rawcache]], %for.body ]


; CHECK-NEXT:   %[[reallocbc:.+]] = bitcast i8* %[[realloccache]] to %class.node**
; CHECK-NEXT:   %[[geptostore:.+]] = getelementptr inbounds %class.node*, %class.node** %[[reallocbc]], i64 %[[preidx]]
; CHECK-NEXT:   store %class.node* %[[cur]], %class.node** %[[geptostore]]
; CHECK-NEXT:   %[[nextipg:.+]] = getelementptr inbounds %class.node, %class.node* %[[cur]], i64 0, i32 1
; CHECK-NEXT:   %next = getelementptr inbounds %class.node, %class.node* %val.08, i64 0, i32 1
; CHECK-NEXT:   %"'ipl" = load %class.node*, %class.node** %[[nextipg]], align 8
; CHECK-NEXT:   %[[nextload:.+]] = load %class.node*, %class.node** %next, align 8, !tbaa !8
; CHECK-NEXT:   %[[lcmp:.+]] = icmp eq %class.node* %[[nextload]], null
; CHECK-NEXT:   br i1 %[[lcmp]], label %[[antiloop:.+]], label %for.body

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertfor.body.preheader:
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[realloccache]])
; CHECK-NEXT:   br label %invertentry

; CHECK: [[antiloop]]:
; CHECK-NEXT:   %[[antivar:.+]] = phi i64 [ %[[subidx:.+]], %incinvertfor.body ], [ %[[preidx]], %[[mergeblk]] ]
; CHECK-NEXT:   %[[structptr:.+]] = getelementptr inbounds %class.node*, %class.node** %[[reallocbc]], i64 %[[antivar]]
; CHECK-NEXT:   %[[struct:.+]] = load %class.node*, %class.node** %[[structptr]]
; CHECK-NEXT:   %[[valueipg:.+]] = getelementptr inbounds %class.node, %class.node* %[[struct]], i64 0, i32 0
; CHECK-NEXT:   %[[val0:.+]] = load double, double* %[[valueipg]]
; CHECK-NEXT:   %[[addval:.+]] = fadd fast double %[[val0]], %[[differet]]
; CHECK-NEXT:   store double %[[addval]], double* %[[valueipg]]
; CHECK-NEXT:   %[[cmpne:.+]] = icmp eq i64 %[[antivar]], 0
; CHECK-NEXT:   br i1 %[[cmpne]], label %invertfor.body.preheader, label %incinvertfor.body

; CHECK: incinvertfor.body:
; CHECK-NEXT:   %[[subidx]] = add nsw i64 %[[antivar]], -1
; CHECK-NEXT:   br label %[[antiloop]]
; CHECK-NEXT: }
