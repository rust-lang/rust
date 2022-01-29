; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

%struct.Gradients = type { {float, float, float}, {float, float, float} }

declare %struct.Gradients @__enzyme_fwddiff({float, float, float} (<4 x float>)*, ...)

define {float, float, float} @square(<4 x float> %x) {
entry:
  %vec = insertelement <4 x float> %x, float 1.0, i32 3
  %sq = fmul <4 x float> %x, %x
  %cb = fmul <4 x float> %sq, %x          
  %id = shufflevector <4 x float> %sq, <4 x float> %cb, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %res1 = extractelement <4 x float> %id, i32 1
  %res2 = extractelement <4 x float> %id, i32 2
  %res3 = extractelement <4 x float> %id, i32 3
  %agg1 = insertvalue {float, float, float} undef, float %res1, 0
  %agg2 = insertvalue {float, float, float} %agg1, float %res2, 1
  %agg3 = insertvalue {float, float, float} %agg2, float %res3, 2
  ret {float, float, float} %agg3
}

define %struct.Gradients @dsquare(<4 x float> %x) {
entry:
  %call = tail call %struct.Gradients ({float, float, float} (<4 x float>)*, ...) @__enzyme_fwddiff({float, float, float} (<4 x float>)* @square, metadata !"enzyme_width", i64 2, <4 x float> %x, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
  ret %struct.Gradients %call
}


; CHECK: define internal [2 x { float, float, float }] @fwddiffe2square(<4 x float> %x, [2 x <4 x float>] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %sq = fmul <4 x float> %x, %x
; CHECK-NEXT:   %0 = extractvalue [2 x <4 x float>] %"x'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x <4 x float>] %"x'", 0
; CHECK-NEXT:   %2 = fmul fast <4 x float> %0, %x
; CHECK-NEXT:   %3 = fmul fast <4 x float> %1, %x
; CHECK-NEXT:   %4 = fadd fast <4 x float> %2, %3
; CHECK-NEXT:   %5 = insertvalue [2 x <4 x float>] undef, <4 x float> %4, 0
; CHECK-NEXT:   %6 = extractvalue [2 x <4 x float>] %"x'", 1
; CHECK-NEXT:   %7 = extractvalue [2 x <4 x float>] %"x'", 1
; CHECK-NEXT:   %8 = fmul fast <4 x float> %6, %x
; CHECK-NEXT:   %9 = fmul fast <4 x float> %7, %x
; CHECK-NEXT:   %10 = fadd fast <4 x float> %8, %9
; CHECK-NEXT:   %11 = insertvalue [2 x <4 x float>] %5, <4 x float> %10, 1
; CHECK-NEXT:   %cb = fmul <4 x float> %sq, %x
; CHECK-NEXT:   %12 = extractvalue [2 x <4 x float>] %11, 0
; CHECK-NEXT:   %13 = extractvalue [2 x <4 x float>] %"x'", 0
; CHECK-NEXT:   %14 = fmul fast <4 x float> %12, %x
; CHECK-NEXT:   %15 = fmul fast <4 x float> %13, %sq
; CHECK-NEXT:   %16 = fadd fast <4 x float> %14, %15
; CHECK-NEXT:   %17 = insertvalue [2 x <4 x float>] undef, <4 x float> %16, 0
; CHECK-NEXT:   %18 = extractvalue [2 x <4 x float>] %11, 1
; CHECK-NEXT:   %19 = extractvalue [2 x <4 x float>] %"x'", 1
; CHECK-NEXT:   %20 = fmul fast <4 x float> %18, %x
; CHECK-NEXT:   %21 = fmul fast <4 x float> %19, %sq
; CHECK-NEXT:   %22 = fadd fast <4 x float> %20, %21
; CHECK-NEXT:   %23 = insertvalue [2 x <4 x float>] %17, <4 x float> %22, 1
; CHECK-NEXT:   %id = shufflevector <4 x float> %sq, <4 x float> %cb, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
; CHECK-NEXT:   %24 = extractvalue [2 x <4 x float>] %11, 0
; CHECK-NEXT:   %25 = extractvalue [2 x <4 x float>] %23, 0
; CHECK-NEXT:   %26 = shufflevector <4 x float> %24, <4 x float> %25, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
; CHECK-NEXT:   %27 = insertvalue [2 x <4 x float>] undef, <4 x float> %26, 0
; CHECK-NEXT:   %28 = extractvalue [2 x <4 x float>] %11, 1
; CHECK-NEXT:   %29 = extractvalue [2 x <4 x float>] %23, 1
; CHECK-NEXT:   %30 = shufflevector <4 x float> %28, <4 x float> %29, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
; CHECK-NEXT:   %31 = insertvalue [2 x <4 x float>] %27, <4 x float> %30, 1
; CHECK-NEXT:   %res1 = extractelement <4 x float> %id, i32 1
; CHECK-NEXT:   %32 = extractvalue [2 x <4 x float>] %31, 0
; CHECK-NEXT:   %33 = extractelement <4 x float> %32, i32 1
; CHECK-NEXT:   %34 = insertvalue [2 x float] undef, float %33, 0
; CHECK-NEXT:   %35 = extractvalue [2 x <4 x float>] %31, 1
; CHECK-NEXT:   %36 = extractelement <4 x float> %35, i32 1
; CHECK-NEXT:   %37 = insertvalue [2 x float] %34, float %36, 1
; CHECK-NEXT:   %res2 = extractelement <4 x float> %id, i32 2
; CHECK-NEXT:   %38 = extractvalue [2 x <4 x float>] %31, 0
; CHECK-NEXT:   %39 = extractelement <4 x float> %38, i32 2
; CHECK-NEXT:   %40 = insertvalue [2 x float] undef, float %39, 0
; CHECK-NEXT:   %41 = extractvalue [2 x <4 x float>] %31, 1
; CHECK-NEXT:   %42 = extractelement <4 x float> %41, i32 2
; CHECK-NEXT:   %43 = insertvalue [2 x float] %40, float %42, 1
; CHECK-NEXT:   %res3 = extractelement <4 x float> %id, i32 3
; CHECK-NEXT:   %44 = extractvalue [2 x <4 x float>] %31, 0
; CHECK-NEXT:   %45 = extractelement <4 x float> %44, i32 3
; CHECK-NEXT:   %46 = insertvalue [2 x float] undef, float %45, 0
; CHECK-NEXT:   %47 = extractvalue [2 x <4 x float>] %31, 1
; CHECK-NEXT:   %48 = extractelement <4 x float> %47, i32 3
; CHECK-NEXT:   %49 = insertvalue [2 x float] %46, float %48, 1
; CHECK-NEXT:   %agg1 = insertvalue { float, float, float } undef, float %res1, 0
; CHECK-NEXT:   %50 = extractvalue [2 x float] %37, 0
; CHECK-NEXT:   %51 = insertvalue { float, float, float } zeroinitializer, float %50, 0
; CHECK-NEXT:   %52 = insertvalue [2 x { float, float, float }] undef, { float, float, float } %51, 0
; CHECK-NEXT:   %53 = extractvalue [2 x float] %37, 1
; CHECK-NEXT:   %54 = insertvalue { float, float, float } zeroinitializer, float %53, 0
; CHECK-NEXT:   %55 = insertvalue [2 x { float, float, float }] %52, { float, float, float } %54, 1
; CHECK-NEXT:   %agg2 = insertvalue { float, float, float } %agg1, float %res2, 1
; CHECK-NEXT:   %56 = extractvalue [2 x { float, float, float }] %55, 0
; CHECK-NEXT:   %57 = extractvalue [2 x float] %43, 0
; CHECK-NEXT:   %58 = insertvalue { float, float, float } %56, float %57, 1
; CHECK-NEXT:   %59 = insertvalue [2 x { float, float, float }] undef, { float, float, float } %58, 0
; CHECK-NEXT:   %60 = extractvalue [2 x { float, float, float }] %55, 1
; CHECK-NEXT:   %61 = extractvalue [2 x float] %43, 1
; CHECK-NEXT:   %62 = insertvalue { float, float, float } %60, float %61, 1
; CHECK-NEXT:   %63 = insertvalue [2 x { float, float, float }] %59, { float, float, float } %62, 1
; CHECK-NEXT:   %64 = extractvalue [2 x { float, float, float }] %63, 0
; CHECK-NEXT:   %65 = extractvalue [2 x float] %49, 0
; CHECK-NEXT:   %66 = insertvalue { float, float, float } %64, float %65, 2
; CHECK-NEXT:   %67 = insertvalue [2 x { float, float, float }] undef, { float, float, float } %66, 0
; CHECK-NEXT:   %68 = extractvalue [2 x { float, float, float }] %63, 1
; CHECK-NEXT:   %69 = extractvalue [2 x float] %49, 1
; CHECK-NEXT:   %70 = insertvalue { float, float, float } %68, float %69, 2
; CHECK-NEXT:   %71 = insertvalue [2 x { float, float, float }] %67, { float, float, float } %70, 1
; CHECK-NEXT:   ret [2 x { float, float, float }] %71
; CHECK-NEXT: }