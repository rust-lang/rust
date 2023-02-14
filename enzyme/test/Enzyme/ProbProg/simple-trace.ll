; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s

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

define void @test() {
entry:
  %mu = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double 0.0, double 1.0)
  %x = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %mu, double 1.0)
  ret void
}

define i8* @generate() {
entry:
  %call = tail call i8* @__enzyme_trace(void ()* @test)
  ret i8* %call
}


; CHECK: define i8* @generate()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call i8* @__enzyme_newtrace()
; CHECK-NEXT:   call void @trace_test(i8* %0)
; CHECK-NEXT:   ret i8* %0
; CHECK-NEXT: }


; CHECK: define internal void @trace_test(i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mu = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.mu = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %mu)
; CHECK-NEXT:   %0 = bitcast double %mu to i64
; CHECK-NEXT:   %1 = inttoptr i64 %0 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double %likelihood.mu, i8* %1, i64 8)
; CHECK-NEXT:   %x = call double @normal(double %mu, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.x = call double @normal_logpdf(double %mu, double 1.000000e+00, double %x)
; CHECK-NEXT:   %2 = bitcast double %x to i64
; CHECK-NEXT:   %3 = inttoptr i64 %2 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.x, i8* %3, i64 8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }