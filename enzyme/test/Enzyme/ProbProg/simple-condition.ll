; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s

@enzyme_condition = global i32 0

@.str = private constant [3 x i8] c"mu\00"
@.str.1 = private constant [2 x i8] c"x\00"

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
declare i8* @__enzyme_trace(void ()*)
declare i8* @__enzyme_condition(void ()*, i32, i8*)

define void @test() {
entry:
  %mu = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double 0.0, double 1.0)
  %x = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %mu, double 1.0)
  ret void
}

define i8* @generate() {
entry:
  %call = call i8* @__enzyme_trace(void ()* @test)
  ret i8* %call
}

define i8* @condition(i8* %trace) {
entry:
  %0 = load i32, i32* @enzyme_condition
  %call = call i8* @__enzyme_condition(void ()* @test, i32 %0, i8* %trace)
  ret i8* %call
}


; CHECK: define i8* @condition(i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i32, i32* @enzyme_condition
; CHECK-NEXT:   %1 = call i8* @__enzyme_newtrace()
; CHECK-NEXT:   call void @condition_test(i8* %trace, i8* %1)
; CHECK-NEXT:   ret i8* %1
; CHECK-NEXT: }

; CHECK: define internal void @condition_test(i8* %observations, i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x.ptr = alloca double
; CHECK-NEXT:   %mu.ptr = alloca double
; CHECK-NEXT:   %has.choice.mu = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.mu, label %condition.mu.with.trace, label %condition.mu.without.trace

; CHECK: condition.mu.with.trace:                          ; preds = %entry
; CHECK-NEXT:   %0 = bitcast double* %mu.ptr to i8*
; CHECK-NEXT:   %mu.size = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8* %0, i64 8)
; CHECK-NEXT:   %from.trace.mu = load double, double* %mu.ptr
; CHECK-NEXT:   br label %entry.cntd

; CHECK: condition.mu.without.trace:                       ; preds = %entry
; CHECK-NEXT:   %sample.mu = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %entry.cntd

; CHECK: entry.cntd:                                       ; preds = %condition.mu.without.trace, %condition.mu.with.trace
; CHECK-NEXT:   %mu = phi double [ %from.trace.mu, %condition.mu.with.trace ], [ %sample.mu, %condition.mu.without.trace ]
; CHECK-NEXT:   %likelihood.mu = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %mu)
; CHECK-NEXT:   %1 = bitcast double %mu to i64
; CHECK-NEXT:   %2 = inttoptr i64 %1 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double %likelihood.mu, i8* %2, i64 8)
; CHECK-NEXT:   %has.choice.x = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.x, label %condition.x.with.trace, label %condition.x.without.trace

; CHECK: condition.x.with.trace:                           ; preds = %entry.cntd
; CHECK-NEXT:   %3 = bitcast double* %x.ptr to i8*
; CHECK-NEXT:   %x.size = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %3, i64 8)
; CHECK-NEXT:   %from.trace.x = load double, double* %x.ptr
; CHECK-NEXT:   br label %entry.cntd.cntd

; CHECK: condition.x.without.trace:                        ; preds = %entry.cntd
; CHECK-NEXT:   %sample.x = call double @normal(double %mu, double 1.000000e+00)
; CHECK-NEXT:   br label %entry.cntd.cntd

; CHECK: entry.cntd.cntd:                                  ; preds = %condition.x.without.trace, %condition.x.with.trace
; CHECK-NEXT:   %x = phi double [ %from.trace.x, %condition.x.with.trace ], [ %sample.x, %condition.x.without.trace ]
; CHECK-NEXT:   %likelihood.x = call double @normal_logpdf(double %mu, double 1.000000e+00, double %x)
; CHECK-NEXT:   %4 = bitcast double %x to i64
; CHECK-NEXT:   %5 = inttoptr i64 %4 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.x, i8* %5, i64 8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }