; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s

@.str = private constant [11 x i8] c"predict, 0\00"
@.str.1 = private constant [2 x i8] c"m\00"
@.str.2 = private constant [2 x i8] c"b\00"

@enzyme_condition = global i32 0

declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)

declare i8* @__enzyme_newtrace()
declare void @__enzyme_freetrace(i8*)
declare i8* @__enzyme_get_trace(i8*, i8*)
declare i64 @__enzyme_get_choice(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_call(i8*, i8*, i8*)
declare void @__enzyme_insert_choice(i8* %trace, i8*, double, i8*, i64)
declare i1 @__enzyme_has_call(i8*, i8*)
declare i1 @__enzyme_has_choice(i8*, i8*)
declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare i8* @__enzyme_trace(double (double*, i32)*, double*, i32)
declare i8* @__enzyme_condition(double (double*, i32)*, double*, i32, i32, i8*)


define double @calculate_loss(double %m, double %b, double* %data, i32 %n) {
entry:
  %cmp19 = icmp sgt i32 %n, 0
  br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %loss.0.lcssa = phi double [ 0.0, %entry ], [ %3, %for.body ]
  ret double %loss.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %loss.021 = phi double [ 0.0, %for.body.preheader ], [ %3, %for.body ]
  %0 = trunc i64 %indvars.iv to i32
  %conv2 = sitofp i32 %0 to double
  %mul1 = fmul double %conv2, %m
  %1 = fadd double %mul1, %b 
  %call = tail call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %1, double 1.0)
  %arrayidx3 = getelementptr inbounds double, double* %data, i64 %indvars.iv
  %2 = load double, double* %arrayidx3
  %sub = fsub double %call, %2
  %mul2 = fmul double %sub, %sub
  %3 = fadd double %mul2, %loss.021
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define double @loss(double* %data, i32 %n) {
entry:
  %call = tail call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double 0.0, double 1.0)
  %call1 = tail call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double 0.0, double 1.0)
  %call2 = tail call double @calculate_loss(double %call, double %call1, double* %data, i32 %n)
  ret double %call2
}

define i8* @condition(double* %data, i32 %n, i8* %trace) {
entry:
  %0 = load i32, i32* @enzyme_condition
  %call = tail call i8* @__enzyme_condition(double (double*, i32)* @loss, double* %data, i32 %n, i32 %0, i8* %trace)
  ret i8* %call
}


; CHECK: define i8* @condition(double* %data, i32 %n, i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i32, i32* @enzyme_condition
; CHECK-NEXT:   %1 = call i8* @__enzyme_newtrace()
; CHECK-NEXT:   %2 = call double @condition_loss(double* %data, i32 %n, i8* %trace, i8* %1)
; CHECK-NEXT:   ret i8* %1
; CHECK-NEXT: }


; CHECK: define internal double @condition_loss(double* %data, i32 %n, i8* %observations, i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call1.ptr = alloca double
; CHECK-NEXT:   %call.ptr = alloca double
; CHECK-NEXT:   %has.choice.call = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.call, label %condition.call.with.trace, label %condition.call.without.trace

; CHECK: condition.call.with.trace:                        ; preds = %entry
; CHECK-NEXT:   %0 = bitcast double* %call.ptr to i8*
; CHECK-NEXT:   %call.size = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %0, i64 8)
; CHECK-NEXT:   %from.trace.call = load double, double* %call.ptr
; CHECK-NEXT:   br label %entry.cntd

; CHECK: condition.call.without.trace:                     ; preds = %entry
; CHECK-NEXT:   %sample.call = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %entry.cntd

; CHECK: entry.cntd:                                       ; preds = %condition.call.without.trace, %condition.call.with.trace
; CHECK-NEXT:   %call = phi double [ %from.trace.call, %condition.call.with.trace ], [ %sample.call, %condition.call.without.trace ]
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %call)
; CHECK-NEXT:   %1 = bitcast double %call to i64
; CHECK-NEXT:   %2 = inttoptr i64 %1 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.call, i8* %2, i64 8)
; CHECK-NEXT:   %has.choice.call1 = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.call1, label %condition.call1.with.trace, label %condition.call1.without.trace

; CHECK: condition.call1.with.trace:                       ; preds = %entry.cntd
; CHECK-NEXT:   %3 = bitcast double* %call1.ptr to i8*
; CHECK-NEXT:   %call1.size = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), i8* %3, i64 8)
; CHECK-NEXT:   %from.trace.call1 = load double, double* %call1.ptr
; CHECK-NEXT:   br label %entry.cntd.cntd

; CHECK: condition.call1.without.trace:                    ; preds = %entry.cntd
; CHECK-NEXT:   %sample.call1 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %entry.cntd.cntd

; CHECK: entry.cntd.cntd:                                  ; preds = %condition.call1.without.trace, %condition.call1.with.trace
; CHECK-NEXT:   %call1 = phi double [ %from.trace.call1, %condition.call1.with.trace ], [ %sample.call1, %condition.call1.without.trace ]
; CHECK-NEXT:   %likelihood.call1 = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %call1)
; CHECK-NEXT:   %4 = bitcast double %call1 to i64
; CHECK-NEXT:   %5 = inttoptr i64 %4 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double %likelihood.call1, i8* %5, i64 8)
; CHECK-NEXT:   %trace1 = call i8* @__enzyme_newtrace()
; CHECK-NEXT:   %has.call.call2 = call i1 @__enzyme_has_call(i8* %observations, i8* nocapture readonly getelementptr inbounds ([21 x i8], [21 x i8]* @0, i32 0, i32 0))
; CHECK-NEXT:   br i1 %has.call.call2, label %condition.call2.with.trace, label %condition.call2.without.trace

; CHECK: condition.call2.with.trace:                       ; preds = %entry.cntd.cntd
; CHECK-NEXT:   %calculate_loss.subtrace = call i8* @__enzyme_get_trace(i8* %observations, i8* nocapture readonly getelementptr inbounds ([21 x i8], [21 x i8]* @0, i32 0, i32 0))
; CHECK-NEXT:   %condition.calculate_loss = call double @condition_calculate_loss(double %call, double %call1, double* %data, i32 %n, i8* %calculate_loss.subtrace, i8* %trace1)
; CHECK-NEXT:   br label %entry.cntd.cntd.cntd

; CHECK: condition.call2.without.trace:                    ; preds = %entry.cntd.cntd
; CHECK-NEXT:   %trace.calculate_loss = call double @condition_calculate_loss(double %call, double %call1, double* %data, i32 %n, i8* null, i8* %trace1)
; CHECK-NEXT:   br label %entry.cntd.cntd.cntd

; CHECK: entry.cntd.cntd.cntd:                             ; preds = %condition.call2.without.trace, %condition.call2.with.trace
; CHECK-NEXT:   %call2 = phi double [ %condition.calculate_loss, %condition.call2.with.trace ], [ %trace.calculate_loss, %condition.call2.without.trace ]
; CHECK-NEXT:   call void @__enzyme_insert_call(i8* %trace, i8* nocapture readonly getelementptr inbounds ([21 x i8], [21 x i8]* @0, i32 0, i32 0), i8* %trace1)
; CHECK-NEXT:   ret double %call2
; CHECK-NEXT: }


; CHECK: define internal double @condition_calculate_loss(double %m, double %b, double* %data, i32 %n, i8* %observations, i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call.ptr = alloca double
; CHECK-NEXT:   %cmp19 = icmp sgt i32 %n, 0
; CHECK-NEXT:   br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

; CHECK: for.body.preheader:                               ; preds = %entry
; CHECK-NEXT:   %wide.trip.count = zext i32 %n to i64
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.body.cntd, %entry
; CHECK-NEXT:   %loss.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %6, %for.body.cntd ]
; CHECK-NEXT:   ret double %loss.0.lcssa

; CHECK: for.body:                                         ; preds = %for.body.cntd, %for.body.preheader
; CHECK-NEXT:   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body.cntd ]
; CHECK-NEXT:   %loss.021 = phi double [ 0.000000e+00, %for.body.preheader ], [ %6, %for.body.cntd ]
; CHECK-NEXT:   %0 = trunc i64 %indvars.iv to i32
; CHECK-NEXT:   %conv2 = sitofp i32 %0 to double
; CHECK-NEXT:   %mul1 = fmul double %conv2, %m
; CHECK-NEXT:   %1 = fadd double %mul1, %b
; CHECK-NEXT:   %has.choice.call = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.call, label %condition.call.with.trace, label %condition.call.without.trace

; CHECK: condition.call.with.trace:                        ; preds = %for.body
; CHECK-NEXT:   %2 = bitcast double* %call.ptr to i8*
; CHECK-NEXT:   %call.size = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i8* %2, i64 8)
; CHECK-NEXT:   %from.trace.call = load double, double* %call.ptr
; CHECK-NEXT:   br label %for.body.cntd

; CHECK: condition.call.without.trace:                     ; preds = %for.body
; CHECK-NEXT:   %sample.call = call double @normal(double %1, double 1.000000e+00)
; CHECK-NEXT:   br label %for.body.cntd

; CHECK: for.body.cntd:                                    ; preds = %condition.call.without.trace, %condition.call.with.trace
; CHECK-NEXT:   %call = phi double [ %from.trace.call, %condition.call.with.trace ], [ %sample.call, %condition.call.without.trace ]
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double %1, double 1.000000e+00, double %call)
; CHECK-NEXT:   %3 = bitcast double %call to i64
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %likelihood.call, i8* %4, i64 8)
; CHECK-NEXT:   %arrayidx3 = getelementptr inbounds double, double* %data, i64 %indvars.iv
; CHECK-NEXT:   %5 = load double, double* %arrayidx3
; CHECK-NEXT:   %sub = fsub double %call, %5
; CHECK-NEXT:   %mul2 = fmul double %sub, %sub
; CHECK-NEXT:   %6 = fadd double %mul2, %loss.021
; CHECK-NEXT:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT:   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
; CHECK-NEXT:   br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
; CHECK-NEXT: }