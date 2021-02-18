; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=inp -o /dev/null | FileCheck %s

@ptr = private unnamed_addr global i64 zeroinitializer, align 1

define double @inp(double %x, <2 x double> %v) {
entry:
  %cstx = bitcast double %x to i64 
  %cstv = bitcast <2 x double> %v to <2 x i64>
  %negx = xor i64 %cstx, -9223372036854775808
  %negv = xor <2 x i64> %cstv, <i64 -9223372036854775808, i64 -9223372036854775808>
  %i = load i64, i64* @ptr, align 4
  %negi = xor i64 %i, -9223372036854775808
  %res = bitcast i64 %negi to double
  ret double %res
}


!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: inp - {[-1]:Float@double} |{[-1]:Float@double}:{} {[-1]:Float@double}:{} 
; CHECK-NEXT: double %x: {[-1]:Float@double}
; CHECK-NEXT: <2 x double> %v: {[-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %cstx = bitcast double %x to i64: {[-1]:Float@double}
; CHECK-NEXT:   %cstv = bitcast <2 x double> %v to <2 x i64>: {[-1]:Float@double}
; CHECK-NEXT:   %negx = xor i64 %cstx, -9223372036854775808: {[-1]:Float@double}
; CHECK-NEXT:   %negv = xor <2 x i64> %cstv, <i64 -9223372036854775808, i64 -9223372036854775808>: {[-1]:Float@double}
; CHECK-NEXT:   %i = load i64, i64* @ptr, align 4: {[-1]:Float@double}
; CHECK-NEXT:   %negi = xor i64 %i, -9223372036854775808: {[-1]:Float@double}
; CHECK-NEXT:   %res = bitcast i64 %negi to double: {[-1]:Float@double}
; CHECK-NEXT:   ret double %res: {}