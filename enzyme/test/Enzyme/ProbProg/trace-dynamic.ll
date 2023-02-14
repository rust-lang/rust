; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s

@.str = private constant [11 x i8] c"predict, 0\00"
@.str.1 = private constant [2 x i8] c"m\00"
@.str.2 = private constant [2 x i8] c"b\00"

@enzyme_interface = global i32 0

declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)

declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare i8* @__enzyme_trace(double (double*, i32)*, double*, i32, i32, i8**)


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
  %cmp19.i = icmp sgt i32 %n, 0
  br i1 %cmp19.i, label %for.body.preheader.i, label %calculate_loss.exit

for.body.preheader.i:                             ; preds = %entry
  %wide.trip.count.i = zext i32 %n to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i = phi i64 [ 0, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
  %loss.021.i = phi double [ 0.0, %for.body.preheader.i ], [ %3, %for.body.i ]
  %0 = trunc i64 %indvars.iv.i to i32
  %conv2.i = sitofp i32 %0 to double
  %mul1 = fmul double %conv2.i, %call
  %1 = fadd double %mul1, %call1
  %call.i = tail call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %1, double 1.0)
  %arrayidx3.i = getelementptr inbounds double, double* %data, i64 %indvars.iv.i
  %2 = load double, double* %arrayidx3.i
  %sub.i = fsub double %call.i, %2
  %mul2 = fmul double %sub.i, %sub.i
  %3 = fadd double %mul2, %loss.021.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %calculate_loss.exit, label %for.body.i

calculate_loss.exit:                              ; preds = %for.body.i, %entry
  %loss.0.lcssa.i = phi double [ 0.0, %entry ], [ %3, %for.body.i ]
  ret double %loss.0.lcssa.i
}

define i8* @generate(double* %data, i32 %n, i8** %interface) {
entry:
  %0 = load i32, i32* @enzyme_interface
  %call = tail call i8* @__enzyme_trace(double (double*, i32)* @loss, double* %data, i32 %n, i32 %0, i8** %interface)
  ret i8* %call
}


; CHECK: define i8* @generate(double* %data, i32 %n, i8** %interface)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = getelementptr inbounds i8*, i8** %interface, i32 4
; CHECK-NEXT:   %1 = load i8*, i8** %0
; CHECK-NEXT:   %new_trace = bitcast i8* %1 to i8* ()*
; CHECK-NEXT:   %2 = load i32, i32* @enzyme_interface
; CHECK-NEXT:   %3 = call i8* %new_trace()
; CHECK-NEXT:   %4 = call double @trace_loss(double* %data, i32 %n, i8** %interface, i8* %3)
; CHECK-NEXT:   ret i8* %3
; CHECK-NEXT: }


; CHECK: define internal double @trace_loss(double* %data, i32 %n, i8** %interface, i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = getelementptr inbounds i8*, i8** %interface, i32 3
; CHECK-NEXT:   %1 = load i8*, i8** %0
; CHECK-NEXT:   %insert_choice = bitcast i8* %1 to void (i8*, i8*, double, i8*, i64)*
; CHECK-NEXT:   %call = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %call)
; CHECK-NEXT:   %2 = bitcast double %call to i64
; CHECK-NEXT:   %3 = inttoptr i64 %2 to i8*
; CHECK-NEXT:   call void %insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.call, i8* %3, i64 8)
; CHECK-NEXT:   %call1 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.call1 = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %call1)
; CHECK-NEXT:   %4 = bitcast double %call1 to i64
; CHECK-NEXT:   %5 = inttoptr i64 %4 to i8*
; CHECK-NEXT:   call void %insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double %likelihood.call1, i8* %5, i64 8)
; CHECK-NEXT:   %cmp19.i = icmp sgt i32 %n, 0
; CHECK-NEXT:   br i1 %cmp19.i, label %for.body.preheader.i, label %calculate_loss.exit

; CHECK: for.body.preheader.i:                             ; preds = %entry
; CHECK-NEXT:   %wide.trip.count.i = zext i32 %n to i64
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
; CHECK-NEXT:   %indvars.iv.i = phi i64 [ 0, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
; CHECK-NEXT:   %loss.021.i = phi double [ 0.000000e+00, %for.body.preheader.i ], [ %11, %for.body.i ]
; CHECK-NEXT:   %6 = trunc i64 %indvars.iv.i to i32
; CHECK-NEXT:   %conv2.i = sitofp i32 %6 to double
; CHECK-NEXT:   %mul1 = fmul double %conv2.i, %call
; CHECK-NEXT:   %7 = fadd double %mul1, %call1
; CHECK-NEXT:   %call.i = call double @normal(double %7, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.call.i = call double @normal_logpdf(double %7, double 1.000000e+00, double %call.i)
; CHECK-NEXT:   %8 = bitcast double %call.i to i64
; CHECK-NEXT:   %9 = inttoptr i64 %8 to i8*
; CHECK-NEXT:   call void %insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %likelihood.call.i, i8* %9, i64 8)
; CHECK-NEXT:   %arrayidx3.i = getelementptr inbounds double, double* %data, i64 %indvars.iv.i
; CHECK-NEXT:   %10 = load double, double* %arrayidx3.i
; CHECK-NEXT:   %sub.i = fsub double %call.i, %10
; CHECK-NEXT:   %mul2 = fmul double %sub.i, %sub.i
; CHECK-NEXT:   %11 = fadd double %mul2, %loss.021.i
; CHECK-NEXT:   %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
; CHECK-NEXT:   %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
; CHECK-NEXT:   br i1 %exitcond.not.i, label %calculate_loss.exit, label %for.body.i

; CHECK: calculate_loss.exit:                              ; preds = %for.body.i, %entry
; CHECK-NEXT:   %loss.0.lcssa.i = phi double [ 0.000000e+00, %entry ], [ %11, %for.body.i ]
; CHECK-NEXT:   ret double %loss.0.lcssa.i
; CHECK-NEXT: }