%0 = type { i32, i32, i32, i32, [0 x i32] }
%task = type { i32, i32, i32, i32, i32, i32, i32, i32 }
%tydesc = type { %tydesc**, i32, i32, void (i1*, %task*, i1*, %tydesc**, i8*)*, void (i1*, %task*, i1*, %tydesc**, i8*)*, void (i1*, %task*, i1*, %tydesc**, i8*)*, void (i1*, %task*, i1*, %tydesc**, i8*)*, void (i1*, %task*, i1*, %tydesc**, i8*)*, void (i1*, %task*, i1*, %tydesc**, i8*)*, void (i1*, %task*, i1*, %tydesc**, i8*)*, void (i1*, %task*, i1*, %tydesc**, i8*, i8*, i8)* }

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define fastcc void @rust_vec_append_glue(%task*, %tydesc*, %tydesc*, %0**, %0*, i1) {
allocas:
  %6 = alloca i32
  %7 = alloca i8*
  br label %copy_args

copy_args:                                        ; preds = %allocas
  br label %derived_tydescs

derived_tydescs:                                  ; preds = %copy_args
  br label %8

; <label>:8                                       ; preds = %derived_tydescs
  %9 = load %0** %3
  %10 = getelementptr %0* %4, i32 0, i32 2
  %11 = load i32* %10
  %12 = sub i32 %11, 1
  %13 = select i1 %5, i32 %12, i32 %11
  %14 = call %0* @upcall_vec_grow(%task* %0, %0* %9, i32 %13, i32* %6, %tydesc* %1)
  %15 = getelementptr %0* %14, i32 0, i32 2
  %16 = load i32* %15
  %17 = sub i32 %16, 1
  %18 = select i1 %5, i32 %17, i32 %16
  %19 = getelementptr %0* %14, i32 0, i32 4
  %20 = bitcast [0 x i32]* %19 to i8*
  %21 = getelementptr i8* %20, i32 %18
  store i8* %21, i8** %7
  %22 = load i32* %6
  %23 = trunc i32 %22 to i1
  br i1 %23, label %24, label %41

; <label>:24                                      ; preds = %8
  %25 = getelementptr %0* %9, i32 0, i32 2
  %26 = load i32* %25
  %27 = sub i32 %26, 1
  %28 = select i1 %5, i32 %27, i32 %26
  %29 = getelementptr %0* %14, i32 0, i32 4
  %30 = bitcast [0 x i32]* %29 to i8*
  %31 = getelementptr %0* %9, i32 0, i32 4
  %32 = bitcast [0 x i32]* %31 to i8*
  %33 = getelementptr i8* %32, i32 %28
  %34 = getelementptr %tydesc* %2, i32 0, i32 1
  %35 = load i32* %34
  %36 = getelementptr %tydesc* %2, i32 0, i32 2
  %37 = load i32* %36
  %38 = ptrtoint i8* %30 to i32
  %39 = ptrtoint i8* %32 to i32
  %40 = ptrtoint i8* %33 to i32
  br label %55

; <label>:41                                      ; preds = %68, %8
  %42 = getelementptr %0* %4, i32 0, i32 2
  %43 = load i32* %42
  %44 = load i8** %7
  %45 = getelementptr %0* %4, i32 0, i32 4
  %46 = bitcast [0 x i32]* %45 to i8*
  %47 = getelementptr i8* %46, i32 %43
  %48 = getelementptr %tydesc* %2, i32 0, i32 1
  %49 = load i32* %48
  %50 = getelementptr %tydesc* %2, i32 0, i32 2
  %51 = load i32* %50
  %52 = ptrtoint i8* %44 to i32
  %53 = ptrtoint i8* %46 to i32
  %54 = ptrtoint i8* %47 to i32
  br label %79

; <label>:55                                      ; preds = %59, %24
  %56 = phi i32 [ %38, %24 ], [ %66, %59 ]
  %57 = phi i32 [ %39, %24 ], [ %67, %59 ]
  %58 = icmp ult i32 %57, %40
  br i1 %58, label %59, label %68

; <label>:59                                      ; preds = %55
  %60 = inttoptr i32 %56 to i8*
  %61 = inttoptr i32 %57 to i8*
  %62 = getelementptr %tydesc* %2, i32 0, i32 0
  %63 = load %tydesc*** %62
  %64 = getelementptr %tydesc* %2, i32 0, i32 3
  %65 = load void (i1*, %task*, i1*, %tydesc**, i8*)** %64
  call fastcc void %65(i1* null, %task* %0, i1* null, %tydesc** %63, i8* %61)
  %66 = add i32 %56, %35
  %67 = add i32 %57, %35
  br label %55

; <label>:68                                      ; preds = %55
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %30, i8* %32, i32 %28, i32 0, i1 false)
  %69 = getelementptr %0* %9, i32 0, i32 2
  %70 = load i32* %69
  %71 = getelementptr %0* %14, i32 0, i32 2
  store i32 %70, i32* %71
  %72 = getelementptr %0* %14, i32 0, i32 2
  %73 = load i32* %72
  %74 = sub i32 %73, 1
  %75 = select i1 %5, i32 %74, i32 %73
  %76 = getelementptr %0* %14, i32 0, i32 4
  %77 = bitcast [0 x i32]* %76 to i8*
  %78 = getelementptr i8* %77, i32 %75
  store i8* %78, i8** %7
  br label %41

; <label>:79                                      ; preds = %83, %41
  %80 = phi i32 [ %52, %41 ], [ %90, %83 ]
  %81 = phi i32 [ %53, %41 ], [ %91, %83 ]
  %82 = icmp ult i32 %81, %54
  br i1 %82, label %83, label %92

; <label>:83                                      ; preds = %79
  %84 = inttoptr i32 %80 to i8*
  %85 = inttoptr i32 %81 to i8*
  %86 = getelementptr %tydesc* %2, i32 0, i32 0
  %87 = load %tydesc*** %86
  %88 = getelementptr %tydesc* %2, i32 0, i32 3
  %89 = load void (i1*, %task*, i1*, %tydesc**, i8*)** %88
  call fastcc void %89(i1* null, %task* %0, i1* null, %tydesc** %87, i8* %85)
  %90 = add i32 %80, %49
  %91 = add i32 %81, %49
  br label %79

; <label>:92                                      ; preds = %79
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %44, i8* %46, i32 %43, i32 0, i1 false)
  %93 = getelementptr %0* %14, i32 0, i32 2
  %94 = load i32* %93
  %95 = sub i32 %94, 1
  %96 = select i1 %5, i32 %95, i32 %94
  %97 = add i32 %96, %43
  %98 = getelementptr %0* %14, i32 0, i32 2
  store i32 %97, i32* %98
  store %0* %14, %0** %3
  ret void
}

declare %0* @upcall_vec_grow(%task*, %0*, i32, i32*, %tydesc*)
