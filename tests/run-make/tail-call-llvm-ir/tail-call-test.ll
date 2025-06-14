; ModuleID = 'tail_call_test.7c02b6d4e63aa8f0-cgu.0'
source_filename = "tail_call_test.7c02b6d4e63aa8f0-cgu.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx11.0.0"

%"core::fmt::rt::Argument<'_>" = type { %"core::fmt::rt::ArgumentType<'_>" }
%"core::fmt::rt::ArgumentType<'_>" = type { ptr, [1 x i64] }

@vtable.0 = private constant <{ [24 x i8], ptr, ptr, ptr }> <{ [24 x i8] c"\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00", ptr @"_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h4fa86179f7747905E", ptr @"_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h9631b88370906e50E", ptr @"_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h9631b88370906e50E" }>, align 8, !dbg !0
@anon.8f6558de37f42f53bff2af1f8e7dc9d8.0 = private unnamed_addr constant <{ [8 x i8], [8 x i8] }> <{ [8 x i8] zeroinitializer, [8 x i8] undef }>, align 8
@alloc_73a0fade79f6dec35d3a164188d3f328 = private unnamed_addr constant <{ [17 x i8] }> <{ [17 x i8] c"tail-call-test.rs" }>, align 1
@alloc_5813a272eb42ea3a81a9088a043c1814 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_73a0fade79f6dec35d3a164188d3f328, [16 x i8] c"\11\00\00\00\00\00\00\00\09\00\00\00\1A\00\00\00" }>, align 8
@alloc_701437b71fb12e826cca45d25bf54394 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_73a0fade79f6dec35d3a164188d3f328, [16 x i8] c"\11\00\00\00\00\00\00\00\13\00\00\00\11\00\00\00" }>, align 8
@alloc_24e0902b97ab05744658dee33cfad6bd = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_73a0fade79f6dec35d3a164188d3f328, [16 x i8] c"\11\00\00\00\00\00\00\00\1C\00\00\00#\00\00\00" }>, align 8
@alloc_381b85d74f28f9e668d3c070c2e3af56 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_73a0fade79f6dec35d3a164188d3f328, [16 x i8] c"\11\00\00\00\00\00\00\00#\00\00\00$\00\00\00" }>, align 8
@alloc_8f8fd5dbae8400c38e1639b7ea801d16 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_73a0fade79f6dec35d3a164188d3f328, [16 x i8] c"\11\00\00\00\00\00\00\00,\00\00\00\1A\00\00\00" }>, align 8
@alloc_911aa589553e6649ae5da752650f3354 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_73a0fade79f6dec35d3a164188d3f328, [16 x i8] c"\11\00\00\00\00\00\00\003\00\00\00\1B\00\00\00" }>, align 8
@alloc_932e95a4c1bb019d25afba309491f49d = private unnamed_addr constant <{ [9 x i8] }> <{ [9 x i8] c"Results: " }>, align 1
@alloc_0242e8ee118de705af76c627590b82cc = private unnamed_addr constant <{ [1 x i8] }> <{ [1 x i8] c" " }>, align 1
@alloc_49a1e817e911805af64bbc7efb390101 = private unnamed_addr constant <{ [1 x i8] }> <{ [1 x i8] c"\0A" }>, align 1
@alloc_70b8f01ca3944289d1c53010fd7cc68a = private unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8] }> <{ ptr @alloc_932e95a4c1bb019d25afba309491f49d, [8 x i8] c"\09\00\00\00\00\00\00\00", ptr @alloc_0242e8ee118de705af76c627590b82cc, [8 x i8] c"\01\00\00\00\00\00\00\00", ptr @alloc_0242e8ee118de705af76c627590b82cc, [8 x i8] c"\01\00\00\00\00\00\00\00", ptr @alloc_0242e8ee118de705af76c627590b82cc, [8 x i8] c"\01\00\00\00\00\00\00\00", ptr @alloc_0242e8ee118de705af76c627590b82cc, [8 x i8] c"\01\00\00\00\00\00\00\00", ptr @alloc_0242e8ee118de705af76c627590b82cc, [8 x i8] c"\01\00\00\00\00\00\00\00", ptr @alloc_49a1e817e911805af64bbc7efb390101, [8 x i8] c"\01\00\00\00\00\00\00\00" }>, align 8

; std::rt::lang_start
; Function Attrs: uwtable
define hidden i64 @_ZN3std2rt10lang_start17h73b0c5d8223deb96E(ptr %main, i64 %argc, ptr %argv, i8 %sigpipe) unnamed_addr #0 !dbg !45 {
start:
  %sigpipe.dbg.spill = alloca [1 x i8], align 1
  %argv.dbg.spill = alloca [8 x i8], align 8
  %argc.dbg.spill = alloca [8 x i8], align 8
  %main.dbg.spill = alloca [8 x i8], align 8
  %_7 = alloca [8 x i8], align 8
  store ptr %main, ptr %main.dbg.spill, align 8
    #dbg_declare(ptr %main.dbg.spill, !53, !DIExpression(), !59)
  store i64 %argc, ptr %argc.dbg.spill, align 8
    #dbg_declare(ptr %argc.dbg.spill, !54, !DIExpression(), !60)
  store ptr %argv, ptr %argv.dbg.spill, align 8
    #dbg_declare(ptr %argv.dbg.spill, !55, !DIExpression(), !61)
  store i8 %sigpipe, ptr %sigpipe.dbg.spill, align 1
    #dbg_declare(ptr %sigpipe.dbg.spill, !56, !DIExpression(), !62)
  store ptr %main, ptr %_7, align 8, !dbg !63
; call std::rt::lang_start_internal
  %_0 = call i64 @_ZN3std2rt19lang_start_internal17hff478237ff51a9c7E(ptr align 1 %_7, ptr align 8 @vtable.0, i64 %argc, ptr %argv, i8 %sigpipe), !dbg !64
  ret i64 %_0, !dbg !65
}

; std::rt::lang_start::{{closure}}
; Function Attrs: inlinehint uwtable
define internal i32 @"_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h9631b88370906e50E"(ptr align 8 %_1) unnamed_addr #1 !dbg !66 {
start:
  %self.dbg.spill = alloca [1 x i8], align 1
  %_1.dbg.spill = alloca [8 x i8], align 8
  store ptr %_1, ptr %_1.dbg.spill, align 8
    #dbg_declare(ptr %_1.dbg.spill, !72, !DIExpression(DW_OP_deref), !73)
  %_4 = load ptr, ptr %_1, align 8, !dbg !74
; call std::sys::backtrace::__rust_begin_short_backtrace
  call void @_ZN3std3sys9backtrace28__rust_begin_short_backtrace17hac99c3a0f565fd14E(ptr %_4), !dbg !75
; call <() as std::process::Termination>::report
  %self = call i8 @"_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17hccf942a05a23072aE"(), !dbg !75
  store i8 %self, ptr %self.dbg.spill, align 1, !dbg !75
    #dbg_declare(ptr %self.dbg.spill, !76, !DIExpression(), !95)
  %_0 = zext i8 %self to i32, !dbg !97
  ret i32 %_0, !dbg !105
}

; std::sys::backtrace::__rust_begin_short_backtrace
; Function Attrs: noinline uwtable
define internal void @_ZN3std3sys9backtrace28__rust_begin_short_backtrace17hac99c3a0f565fd14E(ptr %f) unnamed_addr #2 !dbg !106 {
start:
  %dummy.dbg.spill = alloca [0 x i8], align 1
  %f.dbg.spill = alloca [8 x i8], align 8
  %result.dbg.spill = alloca [0 x i8], align 1
    #dbg_declare(ptr %result.dbg.spill, !113, !DIExpression(), !117)
  store ptr %f, ptr %f.dbg.spill, align 8
    #dbg_declare(ptr %f.dbg.spill, !112, !DIExpression(), !118)
    #dbg_declare(ptr %dummy.dbg.spill, !119, !DIExpression(), !126)
; call core::ops::function::FnOnce::call_once
  call void @_ZN4core3ops8function6FnOnce9call_once17hfd4f30177a814130E(ptr %f), !dbg !128
  call void asm sideeffect "", "~{memory}"(), !dbg !129, !srcloc !130
  ret void, !dbg !131
}

; core::fmt::rt::Argument::new_display
; Function Attrs: inlinehint uwtable
define internal void @_ZN4core3fmt2rt8Argument11new_display17h4175f487a240089cE(ptr sret([16 x i8]) align 8 %_0, ptr align 1 %x) unnamed_addr #1 !dbg !132 {
start:
  %x.dbg.spill = alloca [8 x i8], align 8
  %_3 = alloca [16 x i8], align 8
  store ptr %x, ptr %x.dbg.spill, align 8
    #dbg_declare(ptr %x.dbg.spill, !243, !DIExpression(), !244)
    #dbg_declare(ptr %x.dbg.spill, !245, !DIExpression(), !254)
    #dbg_declare(ptr %x.dbg.spill, !256, !DIExpression(), !267)
  store ptr %x, ptr %_3, align 8, !dbg !269
  %0 = getelementptr inbounds i8, ptr %_3, i64 8, !dbg !269
  store ptr @"_ZN43_$LT$bool$u20$as$u20$core..fmt..Display$GT$3fmt17hd249e77bcd60ee87E", ptr %0, align 8, !dbg !269
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_0, ptr align 8 %_3, i64 16, i1 false), !dbg !270
  ret void, !dbg !271
}

; core::fmt::rt::Argument::new_display
; Function Attrs: inlinehint uwtable
define internal void @_ZN4core3fmt2rt8Argument11new_display17ha8da45a5453b2abdE(ptr sret([16 x i8]) align 8 %_0, ptr align 4 %x) unnamed_addr #1 !dbg !272 {
start:
  %x.dbg.spill = alloca [8 x i8], align 8
  %_3 = alloca [16 x i8], align 8
  store ptr %x, ptr %x.dbg.spill, align 8
    #dbg_declare(ptr %x.dbg.spill, !280, !DIExpression(), !281)
    #dbg_declare(ptr %x.dbg.spill, !282, !DIExpression(), !291)
    #dbg_declare(ptr %x.dbg.spill, !293, !DIExpression(), !303)
  store ptr %x, ptr %_3, align 8, !dbg !305
  %0 = getelementptr inbounds i8, ptr %_3, i64 8, !dbg !305
  store ptr @"_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u32$GT$3fmt17h2d6296303bc86543E", ptr %0, align 8, !dbg !305
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_0, ptr align 8 %_3, i64 16, i1 false), !dbg !306
  ret void, !dbg !307
}

; core::fmt::Arguments::new_v1
; Function Attrs: inlinehint uwtable
define internal void @_ZN4core3fmt9Arguments6new_v117h7b06876a57e3281cE(ptr sret([48 x i8]) align 8 %_0, ptr align 8 %pieces, ptr align 8 %args) unnamed_addr #1 !dbg !308 {
start:
  %args.dbg.spill = alloca [8 x i8], align 8
  %pieces.dbg.spill = alloca [8 x i8], align 8
  store ptr %pieces, ptr %pieces.dbg.spill, align 8
    #dbg_declare(ptr %pieces.dbg.spill, !381, !DIExpression(), !383)
  store ptr %args, ptr %args.dbg.spill, align 8
    #dbg_declare(ptr %args.dbg.spill, !382, !DIExpression(), !384)
  store ptr %pieces, ptr %_0, align 8, !dbg !385
  %0 = getelementptr inbounds i8, ptr %_0, i64 8, !dbg !385
  store i64 7, ptr %0, align 8, !dbg !385
  %1 = load ptr, ptr @anon.8f6558de37f42f53bff2af1f8e7dc9d8.0, align 8, !dbg !385
  %2 = load i64, ptr getelementptr inbounds (i8, ptr @anon.8f6558de37f42f53bff2af1f8e7dc9d8.0, i64 8), align 8, !dbg !385
  %3 = getelementptr inbounds i8, ptr %_0, i64 32, !dbg !385
  store ptr %1, ptr %3, align 8, !dbg !385
  %4 = getelementptr inbounds i8, ptr %3, i64 8, !dbg !385
  store i64 %2, ptr %4, align 8, !dbg !385
  %5 = getelementptr inbounds i8, ptr %_0, i64 16, !dbg !385
  store ptr %args, ptr %5, align 8, !dbg !385
  %6 = getelementptr inbounds i8, ptr %5, i64 8, !dbg !385
  store i64 6, ptr %6, align 8, !dbg !385
  ret void, !dbg !386
}

; core::ops::function::FnOnce::call_once{{vtable.shim}}
; Function Attrs: inlinehint uwtable
define internal i32 @"_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h4fa86179f7747905E"(ptr %_1) unnamed_addr #1 !dbg !387 {
start:
  %_1.dbg.spill = alloca [8 x i8], align 8
  %_2 = alloca [0 x i8], align 1
  store ptr %_1, ptr %_1.dbg.spill, align 8
    #dbg_declare(ptr %_1.dbg.spill, !396, !DIExpression(), !401)
    #dbg_declare(ptr %_2, !397, !DIExpression(), !401)
  %0 = load ptr, ptr %_1, align 8, !dbg !401
; call core::ops::function::FnOnce::call_once
  %_0 = call i32 @_ZN4core3ops8function6FnOnce9call_once17ha95bcc209a0f6c7cE(ptr %0), !dbg !401
  ret i32 %_0, !dbg !401
}

; core::ops::function::FnOnce::call_once
; Function Attrs: inlinehint uwtable
define internal i32 @_ZN4core3ops8function6FnOnce9call_once17ha95bcc209a0f6c7cE(ptr %0) unnamed_addr #1 personality ptr @rust_eh_personality !dbg !402 {
start:
  %1 = alloca [16 x i8], align 8
  %_2 = alloca [0 x i8], align 1
  %_1 = alloca [8 x i8], align 8
  store ptr %0, ptr %_1, align 8
    #dbg_declare(ptr %_1, !406, !DIExpression(), !408)
    #dbg_declare(ptr %_2, !407, !DIExpression(), !408)
; invoke std::rt::lang_start::{{closure}}
  %_0 = invoke i32 @"_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h9631b88370906e50E"(ptr align 8 %_1)
          to label %bb1 unwind label %cleanup, !dbg !408

bb3:                                              ; preds = %cleanup
  %2 = load ptr, ptr %1, align 8, !dbg !408
  %3 = getelementptr inbounds i8, ptr %1, i64 8, !dbg !408
  %4 = load i32, ptr %3, align 8, !dbg !408
  %5 = insertvalue { ptr, i32 } poison, ptr %2, 0, !dbg !408
  %6 = insertvalue { ptr, i32 } %5, i32 %4, 1, !dbg !408
  resume { ptr, i32 } %6, !dbg !408

cleanup:                                          ; preds = %start
  %7 = landingpad { ptr, i32 }
          cleanup
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = extractvalue { ptr, i32 } %7, 1
  store ptr %8, ptr %1, align 8
  %10 = getelementptr inbounds i8, ptr %1, i64 8
  store i32 %9, ptr %10, align 8
  br label %bb3

bb1:                                              ; preds = %start
  ret i32 %_0, !dbg !408
}

; core::ops::function::FnOnce::call_once
; Function Attrs: inlinehint uwtable
define internal void @_ZN4core3ops8function6FnOnce9call_once17hfd4f30177a814130E(ptr %_1) unnamed_addr #1 !dbg !409 {
start:
  %_1.dbg.spill = alloca [8 x i8], align 8
  %_2 = alloca [0 x i8], align 1
  store ptr %_1, ptr %_1.dbg.spill, align 8
    #dbg_declare(ptr %_1.dbg.spill, !411, !DIExpression(), !415)
    #dbg_declare(ptr %_2, !412, !DIExpression(), !415)
  call void %_1(), !dbg !415
  ret void, !dbg !415
}

; core::ptr::drop_in_place<std::rt::lang_start<()>::{{closure}}>
; Function Attrs: inlinehint uwtable
define internal void @"_ZN4core3ptr85drop_in_place$LT$std..rt..lang_start$LT$$LP$$RP$$GT$..$u7b$$u7b$closure$u7d$$u7d$$GT$17h2dc5dcfe3d451b5dE"(ptr align 8 %_1) unnamed_addr #1 !dbg !416 {
start:
  %_1.dbg.spill = alloca [8 x i8], align 8
  store ptr %_1, ptr %_1.dbg.spill, align 8
    #dbg_declare(ptr %_1.dbg.spill, !421, !DIExpression(), !424)
  ret void, !dbg !424
}

; <() as std::process::Termination>::report
; Function Attrs: inlinehint uwtable
define internal i8 @"_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17hccf942a05a23072aE"() unnamed_addr #1 !dbg !425 {
start:
  %_1.dbg.spill = alloca [0 x i8], align 1
    #dbg_declare(ptr %_1.dbg.spill, !430, !DIExpression(), !431)
  ret i8 0, !dbg !432
}

; tail_call_test::with_tail
; Function Attrs: uwtable
define internal i32 @_ZN14tail_call_test9with_tail17hc3aece45eddbb180E(i32 %n) unnamed_addr #0 !dbg !433 {
start:
  %n.dbg.spill = alloca [4 x i8], align 4
  store i32 %n, ptr %n.dbg.spill, align 4
    #dbg_declare(ptr %n.dbg.spill, !439, !DIExpression(), !440)
  %0 = icmp eq i32 %n, 0, !dbg !441
  br i1 %0, label %bb1, label %bb2, !dbg !441

bb1:                                              ; preds = %start
  ret i32 0, !dbg !442

bb2:                                              ; preds = %start
  %_3.0 = sub i32 %n, 1, !dbg !443
  %_3.1 = icmp ult i32 %n, 1, !dbg !443
  br i1 %_3.1, label %panic, label %bb3, !dbg !443

bb3:                                              ; preds = %bb2
; call tail_call_test::with_tail
  %1 = tail call i32 @_ZN14tail_call_test9with_tail17hc3aece45eddbb180E(i32 %_3.0), !dbg !444
  ret i32 %1, !dbg !444

panic:                                            ; preds = %bb2
; call core::panicking::panic_const::panic_const_sub_overflow
  call void @_ZN4core9panicking11panic_const24panic_const_sub_overflow17h3d6828dc280c05eeE(ptr align 8 @alloc_5813a272eb42ea3a81a9088a043c1814) #7, !dbg !443
  unreachable, !dbg !443
}

; tail_call_test::no_tail
; Function Attrs: uwtable
define internal i32 @_ZN14tail_call_test7no_tail17h8d6adcb9d9b56776E(i32 %n) unnamed_addr #0 !dbg !445 {
start:
  %n.dbg.spill = alloca [4 x i8], align 4
  %_0 = alloca [4 x i8], align 4
  store i32 %n, ptr %n.dbg.spill, align 4
    #dbg_declare(ptr %n.dbg.spill, !447, !DIExpression(), !448)
  %0 = icmp eq i32 %n, 0, !dbg !449
  br i1 %0, label %bb1, label %bb2, !dbg !449

bb1:                                              ; preds = %start
  store i32 0, ptr %_0, align 4, !dbg !450
  br label %bb4, !dbg !451

bb2:                                              ; preds = %start
  %_3.0 = sub i32 %n, 1, !dbg !452
  %_3.1 = icmp ult i32 %n, 1, !dbg !452
  br i1 %_3.1, label %panic, label %bb3, !dbg !452

bb4:                                              ; preds = %bb3, %bb1
  %1 = load i32, ptr %_0, align 4, !dbg !453
  ret i32 %1, !dbg !453

bb3:                                              ; preds = %bb2
; call tail_call_test::no_tail
  %2 = call i32 @_ZN14tail_call_test7no_tail17h8d6adcb9d9b56776E(i32 %_3.0), !dbg !454
  store i32 %2, ptr %_0, align 4, !dbg !454
  br label %bb4, !dbg !454

panic:                                            ; preds = %bb2
; call core::panicking::panic_const::panic_const_sub_overflow
  call void @_ZN4core9panicking11panic_const24panic_const_sub_overflow17h3d6828dc280c05eeE(ptr align 8 @alloc_701437b71fb12e826cca45d25bf54394) #7, !dbg !452
  unreachable, !dbg !452
}

; tail_call_test::even_with_tail
; Function Attrs: uwtable
define internal zeroext i1 @_ZN14tail_call_test14even_with_tail17h8d1dbeb6515757cfE(i32 %n) unnamed_addr #0 !dbg !455 {
start:
  %n.dbg.spill = alloca [4 x i8], align 4
  store i32 %n, ptr %n.dbg.spill, align 4
    #dbg_declare(ptr %n.dbg.spill, !459, !DIExpression(), !460)
  %0 = icmp eq i32 %n, 0, !dbg !461
  br i1 %0, label %bb2, label %bb1, !dbg !461

bb2:                                              ; preds = %start
  ret i1 true, !dbg !462

bb1:                                              ; preds = %start
  %_3.0 = sub i32 %n, 1, !dbg !463
  %_3.1 = icmp ult i32 %n, 1, !dbg !463
  br i1 %_3.1, label %panic, label %bb3, !dbg !463

bb3:                                              ; preds = %bb1
; call tail_call_test::odd_with_tail
  %1 = tail call zeroext i1 @_ZN14tail_call_test13odd_with_tail17hc95516150aaddcc1E(i32 %_3.0), !dbg !464
  ret i1 %1, !dbg !464

panic:                                            ; preds = %bb1
; call core::panicking::panic_const::panic_const_sub_overflow
  call void @_ZN4core9panicking11panic_const24panic_const_sub_overflow17h3d6828dc280c05eeE(ptr align 8 @alloc_24e0902b97ab05744658dee33cfad6bd) #7, !dbg !463
  unreachable, !dbg !463
}

; tail_call_test::odd_with_tail
; Function Attrs: uwtable
define internal zeroext i1 @_ZN14tail_call_test13odd_with_tail17hc95516150aaddcc1E(i32 %n) unnamed_addr #0 !dbg !465 {
start:
  %n.dbg.spill = alloca [4 x i8], align 4
  store i32 %n, ptr %n.dbg.spill, align 4
    #dbg_declare(ptr %n.dbg.spill, !467, !DIExpression(), !468)
  %0 = icmp eq i32 %n, 0, !dbg !469
  br i1 %0, label %bb2, label %bb1, !dbg !469

bb2:                                              ; preds = %start
  ret i1 false, !dbg !470

bb1:                                              ; preds = %start
  %_3.0 = sub i32 %n, 1, !dbg !471
  %_3.1 = icmp ult i32 %n, 1, !dbg !471
  br i1 %_3.1, label %panic, label %bb3, !dbg !471

bb3:                                              ; preds = %bb1
; call tail_call_test::even_with_tail
  %1 = tail call zeroext i1 @_ZN14tail_call_test14even_with_tail17h8d1dbeb6515757cfE(i32 %_3.0), !dbg !472
  ret i1 %1, !dbg !472

panic:                                            ; preds = %bb1
; call core::panicking::panic_const::panic_const_sub_overflow
  call void @_ZN4core9panicking11panic_const24panic_const_sub_overflow17h3d6828dc280c05eeE(ptr align 8 @alloc_381b85d74f28f9e668d3c070c2e3af56) #7, !dbg !471
  unreachable, !dbg !471
}

; tail_call_test::even_no_tail
; Function Attrs: uwtable
define internal zeroext i1 @_ZN14tail_call_test12even_no_tail17hd3be8e5729868b97E(i32 %n) unnamed_addr #0 !dbg !473 {
start:
  %n.dbg.spill = alloca [4 x i8], align 4
  %_0 = alloca [1 x i8], align 1
  store i32 %n, ptr %n.dbg.spill, align 4
    #dbg_declare(ptr %n.dbg.spill, !475, !DIExpression(), !476)
  %0 = icmp eq i32 %n, 0, !dbg !477
  br i1 %0, label %bb2, label %bb1, !dbg !477

bb2:                                              ; preds = %start
  store i8 1, ptr %_0, align 1, !dbg !478
  br label %bb4, !dbg !478

bb1:                                              ; preds = %start
  %_3.0 = sub i32 %n, 1, !dbg !479
  %_3.1 = icmp ult i32 %n, 1, !dbg !479
  br i1 %_3.1, label %panic, label %bb3, !dbg !479

bb4:                                              ; preds = %bb3, %bb2
  %1 = load i8, ptr %_0, align 1, !dbg !480
  %2 = trunc nuw i8 %1 to i1, !dbg !480
  ret i1 %2, !dbg !480

bb3:                                              ; preds = %bb1
; call tail_call_test::odd_no_tail
  %3 = call zeroext i1 @_ZN14tail_call_test11odd_no_tail17h1399b4ae4de50274E(i32 %_3.0), !dbg !481
  %4 = zext i1 %3 to i8, !dbg !481
  store i8 %4, ptr %_0, align 1, !dbg !481
  br label %bb4, !dbg !481

panic:                                            ; preds = %bb1
; call core::panicking::panic_const::panic_const_sub_overflow
  call void @_ZN4core9panicking11panic_const24panic_const_sub_overflow17h3d6828dc280c05eeE(ptr align 8 @alloc_8f8fd5dbae8400c38e1639b7ea801d16) #7, !dbg !479
  unreachable, !dbg !479
}

; tail_call_test::odd_no_tail
; Function Attrs: uwtable
define internal zeroext i1 @_ZN14tail_call_test11odd_no_tail17h1399b4ae4de50274E(i32 %n) unnamed_addr #0 !dbg !482 {
start:
  %n.dbg.spill = alloca [4 x i8], align 4
  %_0 = alloca [1 x i8], align 1
  store i32 %n, ptr %n.dbg.spill, align 4
    #dbg_declare(ptr %n.dbg.spill, !484, !DIExpression(), !485)
  %0 = icmp eq i32 %n, 0, !dbg !486
  br i1 %0, label %bb2, label %bb1, !dbg !486

bb2:                                              ; preds = %start
  store i8 0, ptr %_0, align 1, !dbg !487
  br label %bb4, !dbg !487

bb1:                                              ; preds = %start
  %_3.0 = sub i32 %n, 1, !dbg !488
  %_3.1 = icmp ult i32 %n, 1, !dbg !488
  br i1 %_3.1, label %panic, label %bb3, !dbg !488

bb4:                                              ; preds = %bb3, %bb2
  %1 = load i8, ptr %_0, align 1, !dbg !489
  %2 = trunc nuw i8 %1 to i1, !dbg !489
  ret i1 %2, !dbg !489

bb3:                                              ; preds = %bb1
; call tail_call_test::even_no_tail
  %3 = call zeroext i1 @_ZN14tail_call_test12even_no_tail17hd3be8e5729868b97E(i32 %_3.0), !dbg !490
  %4 = zext i1 %3 to i8, !dbg !490
  store i8 %4, ptr %_0, align 1, !dbg !490
  br label %bb4, !dbg !490

panic:                                            ; preds = %bb1
; call core::panicking::panic_const::panic_const_sub_overflow
  call void @_ZN4core9panicking11panic_const24panic_const_sub_overflow17h3d6828dc280c05eeE(ptr align 8 @alloc_911aa589553e6649ae5da752650f3354) #7, !dbg !488
  unreachable, !dbg !488
}

; tail_call_test::main
; Function Attrs: uwtable
define internal void @_ZN14tail_call_test4main17h34b192e87b023961E() unnamed_addr #0 !dbg !491 {
start:
  %_22 = alloca [16 x i8], align 8
  %_20 = alloca [16 x i8], align 8
  %_18 = alloca [16 x i8], align 8
  %_16 = alloca [16 x i8], align 8
  %_14 = alloca [16 x i8], align 8
  %_12 = alloca [16 x i8], align 8
  %_11 = alloca [96 x i8], align 8
  %_8 = alloca [48 x i8], align 8
  %odd_no_tail_result = alloca [1 x i8], align 1
  %even_no_tail_result = alloca [1 x i8], align 1
  %odd_with_tail_result = alloca [1 x i8], align 1
  %even_with_tail_result = alloca [1 x i8], align 1
  %no_tail_result = alloca [4 x i8], align 4
  %with_tail_result = alloca [4 x i8], align 4
    #dbg_declare(ptr %with_tail_result, !493, !DIExpression(), !505)
    #dbg_declare(ptr %no_tail_result, !495, !DIExpression(), !506)
    #dbg_declare(ptr %even_with_tail_result, !497, !DIExpression(), !507)
    #dbg_declare(ptr %odd_with_tail_result, !499, !DIExpression(), !508)
    #dbg_declare(ptr %even_no_tail_result, !501, !DIExpression(), !509)
    #dbg_declare(ptr %odd_no_tail_result, !503, !DIExpression(), !510)
; call tail_call_test::with_tail
  %0 = call i32 @_ZN14tail_call_test9with_tail17hc3aece45eddbb180E(i32 5), !dbg !511
  store i32 %0, ptr %with_tail_result, align 4, !dbg !511
; call tail_call_test::no_tail
  %1 = call i32 @_ZN14tail_call_test7no_tail17h8d6adcb9d9b56776E(i32 5), !dbg !512
  store i32 %1, ptr %no_tail_result, align 4, !dbg !512
; call tail_call_test::even_with_tail
  %2 = call zeroext i1 @_ZN14tail_call_test14even_with_tail17h8d1dbeb6515757cfE(i32 10), !dbg !513
  %3 = zext i1 %2 to i8, !dbg !513
  store i8 %3, ptr %even_with_tail_result, align 1, !dbg !513
; call tail_call_test::odd_with_tail
  %4 = call zeroext i1 @_ZN14tail_call_test13odd_with_tail17hc95516150aaddcc1E(i32 10), !dbg !514
  %5 = zext i1 %4 to i8, !dbg !514
  store i8 %5, ptr %odd_with_tail_result, align 1, !dbg !514
; call tail_call_test::even_no_tail
  %6 = call zeroext i1 @_ZN14tail_call_test12even_no_tail17hd3be8e5729868b97E(i32 10), !dbg !515
  %7 = zext i1 %6 to i8, !dbg !515
  store i8 %7, ptr %even_no_tail_result, align 1, !dbg !515
; call tail_call_test::odd_no_tail
  %8 = call zeroext i1 @_ZN14tail_call_test11odd_no_tail17h1399b4ae4de50274E(i32 10), !dbg !516
  %9 = zext i1 %8 to i8, !dbg !516
  store i8 %9, ptr %odd_no_tail_result, align 1, !dbg !516
; call core::fmt::rt::Argument::new_display
  call void @_ZN4core3fmt2rt8Argument11new_display17ha8da45a5453b2abdE(ptr sret([16 x i8]) align 8 %_12, ptr align 4 %with_tail_result), !dbg !517
; call core::fmt::rt::Argument::new_display
  call void @_ZN4core3fmt2rt8Argument11new_display17ha8da45a5453b2abdE(ptr sret([16 x i8]) align 8 %_14, ptr align 4 %no_tail_result), !dbg !517
; call core::fmt::rt::Argument::new_display
  call void @_ZN4core3fmt2rt8Argument11new_display17h4175f487a240089cE(ptr sret([16 x i8]) align 8 %_16, ptr align 1 %even_with_tail_result), !dbg !517
; call core::fmt::rt::Argument::new_display
  call void @_ZN4core3fmt2rt8Argument11new_display17h4175f487a240089cE(ptr sret([16 x i8]) align 8 %_18, ptr align 1 %odd_with_tail_result), !dbg !517
; call core::fmt::rt::Argument::new_display
  call void @_ZN4core3fmt2rt8Argument11new_display17h4175f487a240089cE(ptr sret([16 x i8]) align 8 %_20, ptr align 1 %even_no_tail_result), !dbg !517
; call core::fmt::rt::Argument::new_display
  call void @_ZN4core3fmt2rt8Argument11new_display17h4175f487a240089cE(ptr sret([16 x i8]) align 8 %_22, ptr align 1 %odd_no_tail_result), !dbg !517
  %10 = getelementptr inbounds nuw %"core::fmt::rt::Argument<'_>", ptr %_11, i64 0, !dbg !517
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %10, ptr align 8 %_12, i64 16, i1 false), !dbg !517
  %11 = getelementptr inbounds nuw %"core::fmt::rt::Argument<'_>", ptr %_11, i64 1, !dbg !517
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %11, ptr align 8 %_14, i64 16, i1 false), !dbg !517
  %12 = getelementptr inbounds nuw %"core::fmt::rt::Argument<'_>", ptr %_11, i64 2, !dbg !517
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %12, ptr align 8 %_16, i64 16, i1 false), !dbg !517
  %13 = getelementptr inbounds nuw %"core::fmt::rt::Argument<'_>", ptr %_11, i64 3, !dbg !517
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %13, ptr align 8 %_18, i64 16, i1 false), !dbg !517
  %14 = getelementptr inbounds nuw %"core::fmt::rt::Argument<'_>", ptr %_11, i64 4, !dbg !517
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %14, ptr align 8 %_20, i64 16, i1 false), !dbg !517
  %15 = getelementptr inbounds nuw %"core::fmt::rt::Argument<'_>", ptr %_11, i64 5, !dbg !517
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %15, ptr align 8 %_22, i64 16, i1 false), !dbg !517
; call core::fmt::Arguments::new_v1
  call void @_ZN4core3fmt9Arguments6new_v117h7b06876a57e3281cE(ptr sret([48 x i8]) align 8 %_8, ptr align 8 @alloc_70b8f01ca3944289d1c53010fd7cc68a, ptr align 8 %_11), !dbg !517
; call std::io::stdio::_print
  call void @_ZN3std2io5stdio6_print17h976fe03d511b8afaE(ptr align 8 %_8), !dbg !517
  ret void, !dbg !518
}

; std::rt::lang_start_internal
; Function Attrs: uwtable
declare i64 @_ZN3std2rt19lang_start_internal17hff478237ff51a9c7E(ptr align 1, ptr align 8, i64, ptr, i8) unnamed_addr #0

; <bool as core::fmt::Display>::fmt
; Function Attrs: uwtable
declare zeroext i1 @"_ZN43_$LT$bool$u20$as$u20$core..fmt..Display$GT$3fmt17hd249e77bcd60ee87E"(ptr align 1, ptr align 8) unnamed_addr #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #3

; core::fmt::num::imp::<impl core::fmt::Display for u32>::fmt
; Function Attrs: uwtable
declare zeroext i1 @"_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u32$GT$3fmt17h2d6296303bc86543E"(ptr align 4, ptr align 8) unnamed_addr #0

; Function Attrs: nounwind uwtable
declare i32 @rust_eh_personality(i32, i32, i64, ptr, ptr) unnamed_addr #4

; core::panicking::panic_const::panic_const_sub_overflow
; Function Attrs: cold noinline noreturn uwtable
declare void @_ZN4core9panicking11panic_const24panic_const_sub_overflow17h3d6828dc280c05eeE(ptr align 8) unnamed_addr #5

; std::io::stdio::_print
; Function Attrs: uwtable
declare void @_ZN3std2io5stdio6_print17h976fe03d511b8afaE(ptr align 8) unnamed_addr #0

define i32 @main(i32 %0, ptr %1) unnamed_addr #6 {
top:
  %2 = sext i32 %0 to i64
; call std::rt::lang_start
  %3 = call i64 @_ZN3std2rt10lang_start17h73b0c5d8223deb96E(ptr @_ZN14tail_call_test4main17h34b192e87b023961E, i64 %2, ptr %1, i8 0)
  %4 = trunc i64 %3 to i32
  ret i32 %4
}

attributes #0 = { uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="apple-m1" }
attributes #1 = { inlinehint uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="apple-m1" }
attributes #2 = { noinline uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="apple-m1" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nounwind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="apple-m1" }
attributes #5 = { cold noinline noreturn uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="apple-m1" }
attributes #6 = { "frame-pointer"="non-leaf" "target-cpu"="apple-m1" }
attributes #7 = { noreturn }

!llvm.module.flags = !{!24, !25, !26, !27}
!llvm.ident = !{!28}
!llvm.dbg.cu = !{!29}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "<std::rt::lang_start::{closure_env#0}<()> as core::ops::function::Fn<()>>::{vtable}", scope: null, file: !2, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "<unknown>", directory: "")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "<std::rt::lang_start::{closure_env#0}<()> as core::ops::function::Fn<()>>::{vtable_type}", file: !2, size: 384, align: 64, flags: DIFlagArtificial, elements: !4, vtableHolder: !14, templateParams: !23, identifier: "2f8c94bd20c0aded7d0943cfe7a062da")
!4 = !{!5, !8, !10, !11, !12, !13}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "drop_in_place", scope: !3, file: !2, baseType: !6, size: 64, align: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const ()", baseType: !7, size: 64, align: 64, dwarfAddressSpace: 0)
!7 = !DIBasicType(name: "()", encoding: DW_ATE_unsigned)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "size", scope: !3, file: !2, baseType: !9, size: 64, align: 64, offset: 64)
!9 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "align", scope: !3, file: !2, baseType: !9, size: 64, align: 64, offset: 128)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "__method3", scope: !3, file: !2, baseType: !6, size: 64, align: 64, offset: 192)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "__method4", scope: !3, file: !2, baseType: !6, size: 64, align: 64, offset: 256)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "__method5", scope: !3, file: !2, baseType: !6, size: 64, align: 64, offset: 320)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "{closure_env#0}<()>", scope: !15, file: !2, size: 64, align: 64, elements: !18, templateParams: !23, identifier: "a4895dfcea3453f6cd237ae6cb71c022")
!15 = !DINamespace(name: "lang_start", scope: !16)
!16 = !DINamespace(name: "rt", scope: !17)
!17 = !DINamespace(name: "std", scope: null)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "main", scope: !14, file: !2, baseType: !20, size: 64, align: 64)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "fn()", baseType: !21, size: 64, align: 64, dwarfAddressSpace: 0)
!21 = !DISubroutineType(types: !22)
!22 = !{null}
!23 = !{}
!24 = !{i32 8, !"PIC Level", i32 2}
!25 = !{i32 7, !"PIE Level", i32 2}
!26 = !{i32 7, !"Dwarf Version", i32 4}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{!"rustc version 1.87.0-dev"}
!29 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !30, producer: "clang LLVM (rustc version 1.87.0-dev)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !31, globals: !44, splitDebugInlining: false, nameTableKind: None)
!30 = !DIFile(filename: "tail-call-test.rs/@/tail_call_test.7c02b6d4e63aa8f0-cgu.0", directory: "/Users/mhornicky/rust/tests/run-make/tail-call-llvm-ir")
!31 = !{!32, !40}
!32 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Alignment", scope: !33, file: !2, baseType: !35, size: 8, align: 8, flags: DIFlagEnumClass, elements: !36)
!33 = !DINamespace(name: "fmt", scope: !34)
!34 = !DINamespace(name: "core", scope: null)
!35 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!36 = !{!37, !38, !39}
!37 = !DIEnumerator(name: "Left", value: 0, isUnsigned: true)
!38 = !DIEnumerator(name: "Right", value: 1, isUnsigned: true)
!39 = !DIEnumerator(name: "Center", value: 2, isUnsigned: true)
!40 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Alignment", scope: !41, file: !2, baseType: !35, size: 8, align: 8, flags: DIFlagEnumClass, elements: !42)
!41 = !DINamespace(name: "rt", scope: !33)
!42 = !{!37, !38, !39, !43}
!43 = !DIEnumerator(name: "Unknown", value: 3, isUnsigned: true)
!44 = !{!0}
!45 = distinct !DISubprogram(name: "lang_start<()>", linkageName: "_ZN3std2rt10lang_start17h73b0c5d8223deb96E", scope: !16, file: !46, line: 192, type: !47, scopeLine: 192, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !57, retainedNodes: !52)
!46 = !DIFile(filename: "/Users/mhornicky/rust/library/std/src/rt.rs", directory: "", checksumkind: CSK_MD5, checksum: "5ed61ab28987f8860d5842313c6741b3")
!47 = !DISubroutineType(types: !48)
!48 = !{!49, !20, !49, !50, !35}
!49 = !DIBasicType(name: "isize", size: 64, encoding: DW_ATE_signed)
!50 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const *const u8", baseType: !51, size: 64, align: 64, dwarfAddressSpace: 0)
!51 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u8", baseType: !35, size: 64, align: 64, dwarfAddressSpace: 0)
!52 = !{!53, !54, !55, !56}
!53 = !DILocalVariable(name: "main", arg: 1, scope: !45, file: !46, line: 193, type: !20)
!54 = !DILocalVariable(name: "argc", arg: 2, scope: !45, file: !46, line: 194, type: !49)
!55 = !DILocalVariable(name: "argv", arg: 3, scope: !45, file: !46, line: 195, type: !50)
!56 = !DILocalVariable(name: "sigpipe", arg: 4, scope: !45, file: !46, line: 196, type: !35)
!57 = !{!58}
!58 = !DITemplateTypeParameter(name: "T", type: !7)
!59 = !DILocation(line: 193, column: 5, scope: !45)
!60 = !DILocation(line: 194, column: 5, scope: !45)
!61 = !DILocation(line: 195, column: 5, scope: !45)
!62 = !DILocation(line: 196, column: 5, scope: !45)
!63 = !DILocation(line: 199, column: 10, scope: !45)
!64 = !DILocation(line: 198, column: 5, scope: !45)
!65 = !DILocation(line: 204, column: 2, scope: !45)
!66 = distinct !DISubprogram(name: "{closure#0}<()>", linkageName: "_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h9631b88370906e50E", scope: !15, file: !46, line: 199, type: !67, scopeLine: 199, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !57, retainedNodes: !71)
!67 = !DISubroutineType(types: !68)
!68 = !{!69, !70}
!69 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!70 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&std::rt::lang_start::{closure_env#0}<()>", baseType: !14, size: 64, align: 64, dwarfAddressSpace: 0)
!71 = !{!72}
!72 = !DILocalVariable(name: "main", scope: !66, file: !46, line: 193, type: !20, align: 64)
!73 = !DILocation(line: 193, column: 5, scope: !66)
!74 = !DILocation(line: 199, column: 70, scope: !66)
!75 = !DILocation(line: 199, column: 18, scope: !66)
!76 = !DILocalVariable(name: "self", arg: 1, scope: !77, file: !78, line: 2062, type: !79)
!77 = distinct !DISubprogram(name: "to_i32", linkageName: "_ZN3std7process8ExitCode6to_i3217h2aa505e63ed55e56E", scope: !79, file: !78, line: 2062, type: !91, scopeLine: 2062, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, declaration: !93, retainedNodes: !94)
!78 = !DIFile(filename: "/Users/mhornicky/rust/library/std/src/process.rs", directory: "", checksumkind: CSK_MD5, checksum: "dbc7956c9cbb5ca52b3803dc48bc1d09")
!79 = !DICompositeType(tag: DW_TAG_structure_type, name: "ExitCode", scope: !80, file: !2, size: 8, align: 8, flags: DIFlagPublic, elements: !81, templateParams: !23, identifier: "72dea078999b9e93868f7a94d355d67a")
!80 = !DINamespace(name: "process", scope: !17)
!81 = !{!82}
!82 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !79, file: !2, baseType: !83, size: 8, align: 8, flags: DIFlagPrivate)
!83 = !DICompositeType(tag: DW_TAG_structure_type, name: "ExitCode", scope: !84, file: !2, size: 8, align: 8, flags: DIFlagPublic, elements: !89, templateParams: !23, identifier: "4f89cd34e9814efc996e2d1833a5a873")
!84 = !DINamespace(name: "process_common", scope: !85)
!85 = !DINamespace(name: "process", scope: !86)
!86 = !DINamespace(name: "unix", scope: !87)
!87 = !DINamespace(name: "pal", scope: !88)
!88 = !DINamespace(name: "sys", scope: !17)
!89 = !{!90}
!90 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !83, file: !2, baseType: !35, size: 8, align: 8, flags: DIFlagPrivate)
!91 = !DISubroutineType(types: !92)
!92 = !{!69, !79}
!93 = !DISubprogram(name: "to_i32", linkageName: "_ZN3std7process8ExitCode6to_i3217h2aa505e63ed55e56E", scope: !79, file: !78, line: 2062, type: !91, scopeLine: 2062, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !23)
!94 = !{!76}
!95 = !DILocation(line: 2062, column: 19, scope: !77, inlinedAt: !96)
!96 = !DILocation(line: 199, column: 85, scope: !66)
!97 = !DILocation(line: 631, column: 9, scope: !98, inlinedAt: !104)
!98 = distinct !DISubprogram(name: "as_i32", linkageName: "_ZN3std3sys3pal4unix7process14process_common8ExitCode6as_i3217h7fe5ef8d5d6ab3beE", scope: !83, file: !99, line: 630, type: !100, scopeLine: 630, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, declaration: !103)
!99 = !DIFile(filename: "/Users/mhornicky/rust/library/std/src/sys/pal/unix/process/process_common.rs", directory: "", checksumkind: CSK_MD5, checksum: "53377008eaa564ea7afaa2596ee97e68")
!100 = !DISubroutineType(types: !101)
!101 = !{!69, !102}
!102 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&std::sys::pal::unix::process::process_common::ExitCode", baseType: !83, size: 64, align: 64, dwarfAddressSpace: 0)
!103 = !DISubprogram(name: "as_i32", linkageName: "_ZN3std3sys3pal4unix7process14process_common8ExitCode6as_i3217h7fe5ef8d5d6ab3beE", scope: !83, file: !99, line: 630, type: !100, scopeLine: 630, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !23)
!104 = !DILocation(line: 2063, column: 16, scope: !77, inlinedAt: !96)
!105 = !DILocation(line: 199, column: 93, scope: !66)
!106 = distinct !DISubprogram(name: "__rust_begin_short_backtrace<fn(), ()>", linkageName: "_ZN3std3sys9backtrace28__rust_begin_short_backtrace17hac99c3a0f565fd14E", scope: !108, file: !107, line: 148, type: !109, scopeLine: 148, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !115, retainedNodes: !111)
!107 = !DIFile(filename: "/Users/mhornicky/rust/library/std/src/sys/backtrace.rs", directory: "", checksumkind: CSK_MD5, checksum: "9e30c70624c3cf40238860e740bd696f")
!108 = !DINamespace(name: "backtrace", scope: !88)
!109 = !DISubroutineType(types: !110)
!110 = !{null, !20}
!111 = !{!112, !113}
!112 = !DILocalVariable(name: "f", arg: 1, scope: !106, file: !107, line: 148, type: !20)
!113 = !DILocalVariable(name: "result", scope: !114, file: !107, line: 152, type: !7, align: 8)
!114 = distinct !DILexicalBlock(scope: !106, file: !107, line: 152, column: 5)
!115 = !{!116, !58}
!116 = !DITemplateTypeParameter(name: "F", type: !20)
!117 = !DILocation(line: 152, column: 9, scope: !114)
!118 = !DILocation(line: 148, column: 43, scope: !106)
!119 = !DILocalVariable(name: "dummy", scope: !120, file: !121, line: 476, type: !7, align: 8)
!120 = distinct !DISubprogram(name: "black_box<()>", linkageName: "_ZN4core4hint9black_box17h8534f6bf278d4519E", scope: !122, file: !121, line: 476, type: !123, scopeLine: 476, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !57, retainedNodes: !125)
!121 = !DIFile(filename: "/Users/mhornicky/rust/library/core/src/hint.rs", directory: "", checksumkind: CSK_MD5, checksum: "53e417654697d2a6fdb3b165cec3a4bf")
!122 = !DINamespace(name: "hint", scope: !34)
!123 = !DISubroutineType(types: !124)
!124 = !{null, !7}
!125 = !{!119}
!126 = !DILocation(line: 476, column: 27, scope: !120, inlinedAt: !127)
!127 = !DILocation(line: 155, column: 5, scope: !114)
!128 = !DILocation(line: 152, column: 18, scope: !106)
!129 = !DILocation(line: 477, column: 5, scope: !120, inlinedAt: !127)
!130 = !{i64 8485502464173410}
!131 = !DILocation(line: 158, column: 2, scope: !106)
!132 = distinct !DISubprogram(name: "new_display<bool>", linkageName: "_ZN4core3fmt2rt8Argument11new_display17h4175f487a240089cE", scope: !134, file: !133, line: 117, type: !235, scopeLine: 117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !240, declaration: !239, retainedNodes: !242)
!133 = !DIFile(filename: "/Users/mhornicky/rust/library/core/src/fmt/rt.rs", directory: "", checksumkind: CSK_MD5, checksum: "d6ea0cb069b4dd82e2f5159937043044")
!134 = !DICompositeType(tag: DW_TAG_structure_type, name: "Argument", scope: !41, file: !2, size: 128, align: 64, flags: DIFlagPublic, elements: !135, templateParams: !23, identifier: "58602aa2c593cb801ff091631542096c")
!135 = !{!136}
!136 = !DIDerivedType(tag: DW_TAG_member, name: "ty", scope: !134, file: !2, baseType: !137, size: 128, align: 64, flags: DIFlagPrivate)
!137 = !DICompositeType(tag: DW_TAG_structure_type, name: "ArgumentType", scope: !41, file: !2, size: 128, align: 64, flags: DIFlagPrivate, elements: !138, templateParams: !23, identifier: "dd627857a0a4b5126a3e76cda7e5e252")
!138 = !{!139}
!139 = !DICompositeType(tag: DW_TAG_variant_part, scope: !137, file: !2, size: 128, align: 64, elements: !140, templateParams: !23, identifier: "6c932759f42856361ff813d4bc4a8f25", discriminator: !233)
!140 = !{!141, !229}
!141 = !DIDerivedType(tag: DW_TAG_member, name: "Placeholder", scope: !139, file: !2, baseType: !142, size: 128, align: 64)
!142 = !DICompositeType(tag: DW_TAG_structure_type, name: "Placeholder", scope: !137, file: !2, size: 128, align: 64, flags: DIFlagPrivate, elements: !143, templateParams: !23, identifier: "80dee0a6bd69ed65c77b53727d80f07")
!143 = !{!144, !150, !223}
!144 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !142, file: !2, baseType: !145, size: 64, align: 64, flags: DIFlagPrivate)
!145 = !DICompositeType(tag: DW_TAG_structure_type, name: "NonNull<()>", scope: !146, file: !2, size: 64, align: 64, flags: DIFlagPublic, elements: !148, templateParams: !57, identifier: "522b50db7ed759a870943796bd4bf4ef")
!146 = !DINamespace(name: "non_null", scope: !147)
!147 = !DINamespace(name: "ptr", scope: !34)
!148 = !{!149}
!149 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !145, file: !2, baseType: !6, size: 64, align: 64, flags: DIFlagPrivate)
!150 = !DIDerivedType(tag: DW_TAG_member, name: "formatter", scope: !142, file: !2, baseType: !151, size: 64, align: 64, offset: 64, flags: DIFlagPrivate)
!151 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "unsafe fn(core::ptr::non_null::NonNull<()>, &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error>", baseType: !152, size: 64, align: 64, dwarfAddressSpace: 0)
!152 = !DISubroutineType(types: !153)
!153 = !{!154, !145, !171}
!154 = !DICompositeType(tag: DW_TAG_structure_type, name: "Result<(), core::fmt::Error>", scope: !155, file: !2, size: 8, align: 8, flags: DIFlagPublic, elements: !156, templateParams: !23, identifier: "17271d3ec660ce12aee8618532c0d01b")
!155 = !DINamespace(name: "result", scope: !34)
!156 = !{!157}
!157 = !DICompositeType(tag: DW_TAG_variant_part, scope: !154, file: !2, size: 8, align: 8, elements: !158, templateParams: !23, identifier: "9cad8f3040adeb8dbe6012f7f86f4bd0", discriminator: !170)
!158 = !{!159, !166}
!159 = !DIDerivedType(tag: DW_TAG_member, name: "Ok", scope: !157, file: !2, baseType: !160, size: 8, align: 8, extraData: i8 0)
!160 = !DICompositeType(tag: DW_TAG_structure_type, name: "Ok", scope: !154, file: !2, size: 8, align: 8, flags: DIFlagPublic, elements: !161, templateParams: !163, identifier: "a37c316bb402f538ae3e417e84a747f2")
!161 = !{!162}
!162 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !160, file: !2, baseType: !7, align: 8, offset: 8, flags: DIFlagPublic)
!163 = !{!58, !164}
!164 = !DITemplateTypeParameter(name: "E", type: !165)
!165 = !DICompositeType(tag: DW_TAG_structure_type, name: "Error", scope: !33, file: !2, align: 8, flags: DIFlagPublic, elements: !23, identifier: "ee06b97bde4a910016d9c46d09308354")
!166 = !DIDerivedType(tag: DW_TAG_member, name: "Err", scope: !157, file: !2, baseType: !167, size: 8, align: 8, extraData: i8 1)
!167 = !DICompositeType(tag: DW_TAG_structure_type, name: "Err", scope: !154, file: !2, size: 8, align: 8, flags: DIFlagPublic, elements: !168, templateParams: !163, identifier: "843870027c42a051bd66e4e2437dcb2c")
!168 = !{!169}
!169 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !167, file: !2, baseType: !165, align: 8, offset: 8, flags: DIFlagPublic)
!170 = !DIDerivedType(tag: DW_TAG_member, scope: !154, file: !2, baseType: !35, size: 8, align: 8, flags: DIFlagArtificial)
!171 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&mut core::fmt::Formatter", baseType: !172, size: 64, align: 64, dwarfAddressSpace: 0)
!172 = !DICompositeType(tag: DW_TAG_structure_type, name: "Formatter", scope: !33, file: !2, size: 320, align: 64, flags: DIFlagPublic, elements: !173, templateParams: !23, identifier: "a3f681b15854bb9231b344542d7ebc6f")
!173 = !{!174, !212}
!174 = !DIDerivedType(tag: DW_TAG_member, name: "options", scope: !172, file: !2, baseType: !175, size: 160, align: 32, offset: 128, flags: DIFlagPrivate)
!175 = !DICompositeType(tag: DW_TAG_structure_type, name: "FormattingOptions", scope: !33, file: !2, size: 160, align: 32, flags: DIFlagPublic, elements: !176, templateParams: !23, identifier: "bd4dc7dc318efc33da8742bbcae2b323")
!176 = !{!177, !179, !181, !196, !211}
!177 = !DIDerivedType(tag: DW_TAG_member, name: "flags", scope: !175, file: !2, baseType: !178, size: 32, align: 32, offset: 96, flags: DIFlagPrivate)
!178 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!179 = !DIDerivedType(tag: DW_TAG_member, name: "fill", scope: !175, file: !2, baseType: !180, size: 32, align: 32, flags: DIFlagPrivate)
!180 = !DIBasicType(name: "char", size: 32, encoding: DW_ATE_UTF)
!181 = !DIDerivedType(tag: DW_TAG_member, name: "align", scope: !175, file: !2, baseType: !182, size: 8, align: 8, offset: 128, flags: DIFlagPrivate)
!182 = !DICompositeType(tag: DW_TAG_structure_type, name: "Option<core::fmt::Alignment>", scope: !183, file: !2, size: 8, align: 8, flags: DIFlagPublic, elements: !184, templateParams: !23, identifier: "64192f2e10752a36e12c723698eef876")
!183 = !DINamespace(name: "option", scope: !34)
!184 = !{!185}
!185 = !DICompositeType(tag: DW_TAG_variant_part, scope: !182, file: !2, size: 8, align: 8, elements: !186, templateParams: !23, identifier: "d880d00e6d747b0f1f6e8a9e94eaf5c3", discriminator: !195)
!186 = !{!187, !191}
!187 = !DIDerivedType(tag: DW_TAG_member, name: "None", scope: !185, file: !2, baseType: !188, size: 8, align: 8, extraData: i8 3)
!188 = !DICompositeType(tag: DW_TAG_structure_type, name: "None", scope: !182, file: !2, size: 8, align: 8, flags: DIFlagPublic, elements: !23, templateParams: !189, identifier: "3d08e22b372bde88294a2316f9f57841")
!189 = !{!190}
!190 = !DITemplateTypeParameter(name: "T", type: !32)
!191 = !DIDerivedType(tag: DW_TAG_member, name: "Some", scope: !185, file: !2, baseType: !192, size: 8, align: 8)
!192 = !DICompositeType(tag: DW_TAG_structure_type, name: "Some", scope: !182, file: !2, size: 8, align: 8, flags: DIFlagPublic, elements: !193, templateParams: !189, identifier: "3a9e5713ecba1e91a97a517dafcb7ac3")
!193 = !{!194}
!194 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !192, file: !2, baseType: !32, size: 8, align: 8, flags: DIFlagPublic)
!195 = !DIDerivedType(tag: DW_TAG_member, scope: !182, file: !2, baseType: !35, size: 8, align: 8, flags: DIFlagArtificial)
!196 = !DIDerivedType(tag: DW_TAG_member, name: "width", scope: !175, file: !2, baseType: !197, size: 32, align: 16, offset: 32, flags: DIFlagPrivate)
!197 = !DICompositeType(tag: DW_TAG_structure_type, name: "Option<u16>", scope: !183, file: !2, size: 32, align: 16, flags: DIFlagPublic, elements: !198, templateParams: !23, identifier: "1ea4ad2305a474bff14e1070de2c7469")
!198 = !{!199}
!199 = !DICompositeType(tag: DW_TAG_variant_part, scope: !197, file: !2, size: 32, align: 16, elements: !200, templateParams: !23, identifier: "e457208596f738cb64002441082f677f", discriminator: !210)
!200 = !{!201, !206}
!201 = !DIDerivedType(tag: DW_TAG_member, name: "None", scope: !199, file: !2, baseType: !202, size: 32, align: 16, extraData: i16 0)
!202 = !DICompositeType(tag: DW_TAG_structure_type, name: "None", scope: !197, file: !2, size: 32, align: 16, flags: DIFlagPublic, elements: !23, templateParams: !203, identifier: "58e001d170a24d228c2d92df8933f513")
!203 = !{!204}
!204 = !DITemplateTypeParameter(name: "T", type: !205)
!205 = !DIBasicType(name: "u16", size: 16, encoding: DW_ATE_unsigned)
!206 = !DIDerivedType(tag: DW_TAG_member, name: "Some", scope: !199, file: !2, baseType: !207, size: 32, align: 16, extraData: i16 1)
!207 = !DICompositeType(tag: DW_TAG_structure_type, name: "Some", scope: !197, file: !2, size: 32, align: 16, flags: DIFlagPublic, elements: !208, templateParams: !203, identifier: "85b9858219d97ffc606f853a0846aadf")
!208 = !{!209}
!209 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !207, file: !2, baseType: !205, size: 16, align: 16, offset: 16, flags: DIFlagPublic)
!210 = !DIDerivedType(tag: DW_TAG_member, scope: !197, file: !2, baseType: !205, size: 16, align: 16, flags: DIFlagArtificial)
!211 = !DIDerivedType(tag: DW_TAG_member, name: "precision", scope: !175, file: !2, baseType: !197, size: 32, align: 16, offset: 64, flags: DIFlagPrivate)
!212 = !DIDerivedType(tag: DW_TAG_member, name: "buf", scope: !172, file: !2, baseType: !213, size: 128, align: 64, flags: DIFlagPrivate)
!213 = !DICompositeType(tag: DW_TAG_structure_type, name: "&mut dyn core::fmt::Write", file: !2, size: 128, align: 64, elements: !214, templateParams: !23, identifier: "23120c893b2a1591292eb4af455f96b5")
!214 = !{!215, !218}
!215 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !213, file: !2, baseType: !216, size: 64, align: 64)
!216 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !217, size: 64, align: 64, dwarfAddressSpace: 0)
!217 = !DICompositeType(tag: DW_TAG_structure_type, name: "dyn core::fmt::Write", file: !2, align: 8, elements: !23, identifier: "f765e81d0267a50a4f9fa8652eed8bdb")
!218 = !DIDerivedType(tag: DW_TAG_member, name: "vtable", scope: !213, file: !2, baseType: !219, size: 64, align: 64, offset: 64)
!219 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&[usize; 6]", baseType: !220, size: 64, align: 64, dwarfAddressSpace: 0)
!220 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 384, align: 64, elements: !221)
!221 = !{!222}
!222 = !DISubrange(count: 6, lowerBound: 0)
!223 = !DIDerivedType(tag: DW_TAG_member, name: "_lifetime", scope: !142, file: !2, baseType: !224, align: 8, offset: 128, flags: DIFlagPrivate)
!224 = !DICompositeType(tag: DW_TAG_structure_type, name: "PhantomData<&()>", scope: !225, file: !2, align: 8, flags: DIFlagPublic, elements: !23, templateParams: !226, identifier: "190cbe3de07d70b4ddafe790d0ec47b8")
!225 = !DINamespace(name: "marker", scope: !34)
!226 = !{!227}
!227 = !DITemplateTypeParameter(name: "T", type: !228)
!228 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&()", baseType: !7, size: 64, align: 64, dwarfAddressSpace: 0)
!229 = !DIDerivedType(tag: DW_TAG_member, name: "Count", scope: !139, file: !2, baseType: !230, size: 128, align: 64, extraData: i64 0)
!230 = !DICompositeType(tag: DW_TAG_structure_type, name: "Count", scope: !137, file: !2, size: 128, align: 64, flags: DIFlagPrivate, elements: !231, templateParams: !23, identifier: "665ca354dcab946265bf580a16e938a9")
!231 = !{!232}
!232 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !230, file: !2, baseType: !205, size: 16, align: 16, offset: 64, flags: DIFlagPrivate)
!233 = !DIDerivedType(tag: DW_TAG_member, scope: !137, file: !2, baseType: !234, size: 64, align: 64, flags: DIFlagArtificial)
!234 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!235 = !DISubroutineType(types: !236)
!236 = !{!134, !237}
!237 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&bool", baseType: !238, size: 64, align: 64, dwarfAddressSpace: 0)
!238 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!239 = !DISubprogram(name: "new_display<bool>", linkageName: "_ZN4core3fmt2rt8Argument11new_display17h4175f487a240089cE", scope: !134, file: !133, line: 117, type: !235, scopeLine: 117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !240)
!240 = !{!241}
!241 = !DITemplateTypeParameter(name: "T", type: !238)
!242 = !{!243}
!243 = !DILocalVariable(name: "x", arg: 1, scope: !132, file: !133, line: 117, type: !237)
!244 = !DILocation(line: 117, column: 36, scope: !132)
!245 = !DILocalVariable(name: "x", arg: 1, scope: !246, file: !133, line: 103, type: !237)
!246 = distinct !DISubprogram(name: "new<bool>", linkageName: "_ZN4core3fmt2rt8Argument3new17h6636fff44947e17bE", scope: !134, file: !133, line: 103, type: !247, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !240, declaration: !252, retainedNodes: !253)
!247 = !DISubroutineType(types: !248)
!248 = !{!134, !237, !249}
!249 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "fn(&bool, &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error>", baseType: !250, size: 64, align: 64, dwarfAddressSpace: 0)
!250 = !DISubroutineType(types: !251)
!251 = !{!154, !237, !171}
!252 = !DISubprogram(name: "new<bool>", linkageName: "_ZN4core3fmt2rt8Argument3new17h6636fff44947e17bE", scope: !134, file: !133, line: 103, type: !247, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !240)
!253 = !{!245}
!254 = !DILocation(line: 103, column: 25, scope: !246, inlinedAt: !255)
!255 = !DILocation(line: 118, column: 9, scope: !132)
!256 = !DILocalVariable(name: "r", arg: 1, scope: !257, file: !258, line: 267, type: !237)
!257 = distinct !DISubprogram(name: "from_ref<bool>", linkageName: "_ZN4core3ptr8non_null16NonNull$LT$T$GT$8from_ref17h2c8f69d4b8b086feE", scope: !259, file: !258, line: 267, type: !263, scopeLine: 267, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !240, declaration: !265, retainedNodes: !266)
!258 = !DIFile(filename: "/Users/mhornicky/rust/library/core/src/ptr/non_null.rs", directory: "", checksumkind: CSK_MD5, checksum: "3b3cd84fd90af2705fa6d8309deb8eb9")
!259 = !DICompositeType(tag: DW_TAG_structure_type, name: "NonNull<bool>", scope: !146, file: !2, size: 64, align: 64, flags: DIFlagPublic, elements: !260, templateParams: !240, identifier: "b5f8186ecb714cc37cdb774ddbf3bc6d")
!260 = !{!261}
!261 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !259, file: !2, baseType: !262, size: 64, align: 64, flags: DIFlagPrivate)
!262 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const bool", baseType: !238, size: 64, align: 64, dwarfAddressSpace: 0)
!263 = !DISubroutineType(types: !264)
!264 = !{!259, !237}
!265 = !DISubprogram(name: "from_ref<bool>", linkageName: "_ZN4core3ptr8non_null16NonNull$LT$T$GT$8from_ref17h2c8f69d4b8b086feE", scope: !259, file: !258, line: 267, type: !263, scopeLine: 267, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !240)
!266 = !{!256}
!267 = !DILocation(line: 267, column: 27, scope: !257, inlinedAt: !268)
!268 = !DILocation(line: 108, column: 24, scope: !246, inlinedAt: !255)
!269 = !DILocation(line: 107, column: 17, scope: !246, inlinedAt: !255)
!270 = !DILocation(line: 104, column: 9, scope: !246, inlinedAt: !255)
!271 = !DILocation(line: 119, column: 6, scope: !132)
!272 = distinct !DISubprogram(name: "new_display<u32>", linkageName: "_ZN4core3fmt2rt8Argument11new_display17ha8da45a5453b2abdE", scope: !134, file: !133, line: 117, type: !273, scopeLine: 117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !277, declaration: !276, retainedNodes: !279)
!273 = !DISubroutineType(types: !274)
!274 = !{!134, !275}
!275 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&u32", baseType: !178, size: 64, align: 64, dwarfAddressSpace: 0)
!276 = !DISubprogram(name: "new_display<u32>", linkageName: "_ZN4core3fmt2rt8Argument11new_display17ha8da45a5453b2abdE", scope: !134, file: !133, line: 117, type: !273, scopeLine: 117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !277)
!277 = !{!278}
!278 = !DITemplateTypeParameter(name: "T", type: !178)
!279 = !{!280}
!280 = !DILocalVariable(name: "x", arg: 1, scope: !272, file: !133, line: 117, type: !275)
!281 = !DILocation(line: 117, column: 36, scope: !272)
!282 = !DILocalVariable(name: "x", arg: 1, scope: !283, file: !133, line: 103, type: !275)
!283 = distinct !DISubprogram(name: "new<u32>", linkageName: "_ZN4core3fmt2rt8Argument3new17h43420c6ea6a6e78eE", scope: !134, file: !133, line: 103, type: !284, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !277, declaration: !289, retainedNodes: !290)
!284 = !DISubroutineType(types: !285)
!285 = !{!134, !275, !286}
!286 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "fn(&u32, &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error>", baseType: !287, size: 64, align: 64, dwarfAddressSpace: 0)
!287 = !DISubroutineType(types: !288)
!288 = !{!154, !275, !171}
!289 = !DISubprogram(name: "new<u32>", linkageName: "_ZN4core3fmt2rt8Argument3new17h43420c6ea6a6e78eE", scope: !134, file: !133, line: 103, type: !284, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !277)
!290 = !{!282}
!291 = !DILocation(line: 103, column: 25, scope: !283, inlinedAt: !292)
!292 = !DILocation(line: 118, column: 9, scope: !272)
!293 = !DILocalVariable(name: "r", arg: 1, scope: !294, file: !258, line: 267, type: !275)
!294 = distinct !DISubprogram(name: "from_ref<u32>", linkageName: "_ZN4core3ptr8non_null16NonNull$LT$T$GT$8from_ref17hb973156413efe7b4E", scope: !295, file: !258, line: 267, type: !299, scopeLine: 267, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !277, declaration: !301, retainedNodes: !302)
!295 = !DICompositeType(tag: DW_TAG_structure_type, name: "NonNull<u32>", scope: !146, file: !2, size: 64, align: 64, flags: DIFlagPublic, elements: !296, templateParams: !277, identifier: "567cb10bdd7a4457c0bf8a596630d024")
!296 = !{!297}
!297 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !295, file: !2, baseType: !298, size: 64, align: 64, flags: DIFlagPrivate)
!298 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u32", baseType: !178, size: 64, align: 64, dwarfAddressSpace: 0)
!299 = !DISubroutineType(types: !300)
!300 = !{!295, !275}
!301 = !DISubprogram(name: "from_ref<u32>", linkageName: "_ZN4core3ptr8non_null16NonNull$LT$T$GT$8from_ref17hb973156413efe7b4E", scope: !295, file: !258, line: 267, type: !299, scopeLine: 267, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !277)
!302 = !{!293}
!303 = !DILocation(line: 267, column: 27, scope: !294, inlinedAt: !304)
!304 = !DILocation(line: 108, column: 24, scope: !283, inlinedAt: !292)
!305 = !DILocation(line: 107, column: 17, scope: !283, inlinedAt: !292)
!306 = !DILocation(line: 104, column: 9, scope: !283, inlinedAt: !292)
!307 = !DILocation(line: 119, column: 6, scope: !272)
!308 = distinct !DISubprogram(name: "new_v1<7, 6>", linkageName: "_ZN4core3fmt9Arguments6new_v117h7b06876a57e3281cE", scope: !310, file: !309, line: 608, type: !371, scopeLine: 608, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, declaration: !379, retainedNodes: !380)
!309 = !DIFile(filename: "/Users/mhornicky/rust/library/core/src/fmt/mod.rs", directory: "", checksumkind: CSK_MD5, checksum: "c9bf47d5d3e96a7f07932386791c80c5")
!310 = !DICompositeType(tag: DW_TAG_structure_type, name: "Arguments", scope: !33, file: !2, size: 384, align: 64, flags: DIFlagPublic, elements: !311, templateParams: !23, identifier: "3967edcb025267133814100c7f82be4e")
!311 = !{!312, !323, !365}
!312 = !DIDerivedType(tag: DW_TAG_member, name: "pieces", scope: !310, file: !2, baseType: !313, size: 128, align: 64, flags: DIFlagPrivate)
!313 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[&str]", file: !2, size: 128, align: 64, elements: !314, templateParams: !23, identifier: "4e66b00a376d6af5b8765440fb2839f")
!314 = !{!315, !322}
!315 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !313, file: !2, baseType: !316, size: 64, align: 64)
!316 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !317, size: 64, align: 64, dwarfAddressSpace: 0)
!317 = !DICompositeType(tag: DW_TAG_structure_type, name: "&str", file: !2, size: 128, align: 64, elements: !318, templateParams: !23, identifier: "9277eecd40495f85161460476aacc992")
!318 = !{!319, !321}
!319 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !317, file: !2, baseType: !320, size: 64, align: 64)
!320 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 64, align: 64, dwarfAddressSpace: 0)
!321 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !317, file: !2, baseType: !9, size: 64, align: 64, offset: 64)
!322 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !313, file: !2, baseType: !9, size: 64, align: 64, offset: 64)
!323 = !DIDerivedType(tag: DW_TAG_member, name: "fmt", scope: !310, file: !2, baseType: !324, size: 128, align: 64, offset: 256, flags: DIFlagPrivate)
!324 = !DICompositeType(tag: DW_TAG_structure_type, name: "Option<&[core::fmt::rt::Placeholder]>", scope: !183, file: !2, size: 128, align: 64, flags: DIFlagPublic, elements: !325, templateParams: !23, identifier: "ed58242c0df2208badd502ed37757c5a")
!325 = !{!326}
!326 = !DICompositeType(tag: DW_TAG_variant_part, scope: !324, file: !2, size: 128, align: 64, elements: !327, templateParams: !23, identifier: "8d3685ffdffcd71ba2511f7f5d362c54", discriminator: !364)
!327 = !{!328, !360}
!328 = !DIDerivedType(tag: DW_TAG_member, name: "None", scope: !326, file: !2, baseType: !329, size: 128, align: 64, extraData: i64 0)
!329 = !DICompositeType(tag: DW_TAG_structure_type, name: "None", scope: !324, file: !2, size: 128, align: 64, flags: DIFlagPublic, elements: !23, templateParams: !330, identifier: "cd7a6157dd980ccee6efd2fde0aef1f7")
!330 = !{!331}
!331 = !DITemplateTypeParameter(name: "T", type: !332)
!332 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[core::fmt::rt::Placeholder]", file: !2, size: 128, align: 64, elements: !333, templateParams: !23, identifier: "af56e9c9836ba24dab0e842584254f07")
!333 = !{!334, !359}
!334 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !332, file: !2, baseType: !335, size: 64, align: 64)
!335 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !336, size: 64, align: 64, dwarfAddressSpace: 0)
!336 = !DICompositeType(tag: DW_TAG_structure_type, name: "Placeholder", scope: !41, file: !2, size: 448, align: 64, flags: DIFlagPublic, elements: !337, templateParams: !23, identifier: "6d0152db31a43e1016bc6fe86a449012")
!337 = !{!338, !339, !340, !341, !342, !358}
!338 = !DIDerivedType(tag: DW_TAG_member, name: "position", scope: !336, file: !2, baseType: !9, size: 64, align: 64, offset: 256, flags: DIFlagPublic)
!339 = !DIDerivedType(tag: DW_TAG_member, name: "fill", scope: !336, file: !2, baseType: !180, size: 32, align: 32, offset: 352, flags: DIFlagPublic)
!340 = !DIDerivedType(tag: DW_TAG_member, name: "align", scope: !336, file: !2, baseType: !40, size: 8, align: 8, offset: 384, flags: DIFlagPublic)
!341 = !DIDerivedType(tag: DW_TAG_member, name: "flags", scope: !336, file: !2, baseType: !178, size: 32, align: 32, offset: 320, flags: DIFlagPublic)
!342 = !DIDerivedType(tag: DW_TAG_member, name: "precision", scope: !336, file: !2, baseType: !343, size: 128, align: 64, flags: DIFlagPublic)
!343 = !DICompositeType(tag: DW_TAG_structure_type, name: "Count", scope: !41, file: !2, size: 128, align: 64, flags: DIFlagPublic, elements: !344, templateParams: !23, identifier: "50f5faed6a11385fc6c67885c03dd1e5")
!344 = !{!345}
!345 = !DICompositeType(tag: DW_TAG_variant_part, scope: !343, file: !2, size: 128, align: 64, elements: !346, templateParams: !23, identifier: "2dd91ee8226b96909bc9544648bfe787", discriminator: !357)
!346 = !{!347, !351, !355}
!347 = !DIDerivedType(tag: DW_TAG_member, name: "Is", scope: !345, file: !2, baseType: !348, size: 128, align: 64, extraData: i16 0)
!348 = !DICompositeType(tag: DW_TAG_structure_type, name: "Is", scope: !343, file: !2, size: 128, align: 64, flags: DIFlagPublic, elements: !349, templateParams: !23, identifier: "9390e4bb5c8689a32fe4880004955908")
!349 = !{!350}
!350 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !348, file: !2, baseType: !205, size: 16, align: 16, offset: 16, flags: DIFlagPublic)
!351 = !DIDerivedType(tag: DW_TAG_member, name: "Param", scope: !345, file: !2, baseType: !352, size: 128, align: 64, extraData: i16 1)
!352 = !DICompositeType(tag: DW_TAG_structure_type, name: "Param", scope: !343, file: !2, size: 128, align: 64, flags: DIFlagPublic, elements: !353, templateParams: !23, identifier: "bb7e350dccf5b7cfab34d6a8ae86d600")
!353 = !{!354}
!354 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !352, file: !2, baseType: !9, size: 64, align: 64, offset: 64, flags: DIFlagPublic)
!355 = !DIDerivedType(tag: DW_TAG_member, name: "Implied", scope: !345, file: !2, baseType: !356, size: 128, align: 64, extraData: i16 2)
!356 = !DICompositeType(tag: DW_TAG_structure_type, name: "Implied", scope: !343, file: !2, size: 128, align: 64, flags: DIFlagPublic, elements: !23, identifier: "b76e62f3ed4f40c1e2e7c4f21148df9b")
!357 = !DIDerivedType(tag: DW_TAG_member, scope: !343, file: !2, baseType: !205, size: 16, align: 16, flags: DIFlagArtificial)
!358 = !DIDerivedType(tag: DW_TAG_member, name: "width", scope: !336, file: !2, baseType: !343, size: 128, align: 64, offset: 128, flags: DIFlagPublic)
!359 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !332, file: !2, baseType: !9, size: 64, align: 64, offset: 64)
!360 = !DIDerivedType(tag: DW_TAG_member, name: "Some", scope: !326, file: !2, baseType: !361, size: 128, align: 64)
!361 = !DICompositeType(tag: DW_TAG_structure_type, name: "Some", scope: !324, file: !2, size: 128, align: 64, flags: DIFlagPublic, elements: !362, templateParams: !330, identifier: "147d6e8dd0b06bb7b8562974bbac72ec")
!362 = !{!363}
!363 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !361, file: !2, baseType: !332, size: 128, align: 64, flags: DIFlagPublic)
!364 = !DIDerivedType(tag: DW_TAG_member, scope: !324, file: !2, baseType: !234, size: 64, align: 64, flags: DIFlagArtificial)
!365 = !DIDerivedType(tag: DW_TAG_member, name: "args", scope: !310, file: !2, baseType: !366, size: 128, align: 64, offset: 128, flags: DIFlagPrivate)
!366 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[core::fmt::rt::Argument]", file: !2, size: 128, align: 64, elements: !367, templateParams: !23, identifier: "3e1f378cdfe395f4956f75b2ec174ba6")
!367 = !{!368, !370}
!368 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !366, file: !2, baseType: !369, size: 64, align: 64)
!369 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !134, size: 64, align: 64, dwarfAddressSpace: 0)
!370 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !366, file: !2, baseType: !9, size: 64, align: 64, offset: 64)
!371 = !DISubroutineType(types: !372)
!372 = !{!310, !373, !377}
!373 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&[&str; 7]", baseType: !374, size: 64, align: 64, dwarfAddressSpace: 0)
!374 = !DICompositeType(tag: DW_TAG_array_type, baseType: !317, size: 896, align: 64, elements: !375)
!375 = !{!376}
!376 = !DISubrange(count: 7, lowerBound: 0)
!377 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&[core::fmt::rt::Argument; 6]", baseType: !378, size: 64, align: 64, dwarfAddressSpace: 0)
!378 = !DICompositeType(tag: DW_TAG_array_type, baseType: !134, size: 768, align: 64, elements: !221)
!379 = !DISubprogram(name: "new_v1<7, 6>", linkageName: "_ZN4core3fmt9Arguments6new_v117h7b06876a57e3281cE", scope: !310, file: !309, line: 608, type: !371, scopeLine: 608, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !23)
!380 = !{!381, !382}
!381 = !DILocalVariable(name: "pieces", arg: 1, scope: !308, file: !309, line: 609, type: !373)
!382 = !DILocalVariable(name: "args", arg: 2, scope: !308, file: !309, line: 610, type: !377)
!383 = !DILocation(line: 609, column: 9, scope: !308)
!384 = !DILocation(line: 610, column: 9, scope: !308)
!385 = !DILocation(line: 613, column: 9, scope: !308)
!386 = !DILocation(line: 614, column: 6, scope: !308)
!387 = distinct !DISubprogram(name: "call_once<std::rt::lang_start::{closure_env#0}<()>, ()>", linkageName: "_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h4fa86179f7747905E", scope: !389, file: !388, line: 250, type: !392, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !398, retainedNodes: !395)
!388 = !DIFile(filename: "/Users/mhornicky/rust/library/core/src/ops/function.rs", directory: "", checksumkind: CSK_MD5, checksum: "27f40bbdeb6cc525c0d0d7cf434d92c4")
!389 = !DINamespace(name: "FnOnce", scope: !390)
!390 = !DINamespace(name: "function", scope: !391)
!391 = !DINamespace(name: "ops", scope: !34)
!392 = !DISubroutineType(types: !393)
!393 = !{!69, !394}
!394 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut std::rt::lang_start::{closure_env#0}<()>", baseType: !14, size: 64, align: 64, dwarfAddressSpace: 0)
!395 = !{!396, !397}
!396 = !DILocalVariable(arg: 1, scope: !387, file: !388, line: 250, type: !394)
!397 = !DILocalVariable(arg: 2, scope: !387, file: !388, line: 250, type: !7)
!398 = !{!399, !400}
!399 = !DITemplateTypeParameter(name: "Self", type: !14)
!400 = !DITemplateTypeParameter(name: "Args", type: !7)
!401 = !DILocation(line: 250, column: 5, scope: !387)
!402 = distinct !DISubprogram(name: "call_once<std::rt::lang_start::{closure_env#0}<()>, ()>", linkageName: "_ZN4core3ops8function6FnOnce9call_once17ha95bcc209a0f6c7cE", scope: !389, file: !388, line: 250, type: !403, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !398, retainedNodes: !405)
!403 = !DISubroutineType(types: !404)
!404 = !{!69, !14}
!405 = !{!406, !407}
!406 = !DILocalVariable(arg: 1, scope: !402, file: !388, line: 250, type: !14)
!407 = !DILocalVariable(arg: 2, scope: !402, file: !388, line: 250, type: !7)
!408 = !DILocation(line: 250, column: 5, scope: !402)
!409 = distinct !DISubprogram(name: "call_once<fn(), ()>", linkageName: "_ZN4core3ops8function6FnOnce9call_once17hfd4f30177a814130E", scope: !389, file: !388, line: 250, type: !109, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !413, retainedNodes: !410)
!410 = !{!411, !412}
!411 = !DILocalVariable(arg: 1, scope: !409, file: !388, line: 250, type: !20)
!412 = !DILocalVariable(arg: 2, scope: !409, file: !388, line: 250, type: !7)
!413 = !{!414, !400}
!414 = !DITemplateTypeParameter(name: "Self", type: !20)
!415 = !DILocation(line: 250, column: 5, scope: !409)
!416 = distinct !DISubprogram(name: "drop_in_place<std::rt::lang_start::{closure_env#0}<()>>", linkageName: "_ZN4core3ptr85drop_in_place$LT$std..rt..lang_start$LT$$LP$$RP$$GT$..$u7b$$u7b$closure$u7d$$u7d$$GT$17h2dc5dcfe3d451b5dE", scope: !147, file: !417, line: 523, type: !418, scopeLine: 523, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !422, retainedNodes: !420)
!417 = !DIFile(filename: "/Users/mhornicky/rust/library/core/src/ptr/mod.rs", directory: "", checksumkind: CSK_MD5, checksum: "a5cb8c8d2ea510166b9e600d925000e6")
!418 = !DISubroutineType(types: !419)
!419 = !{null, !394}
!420 = !{!421}
!421 = !DILocalVariable(arg: 1, scope: !416, file: !417, line: 523, type: !394)
!422 = !{!423}
!423 = !DITemplateTypeParameter(name: "T", type: !14)
!424 = !DILocation(line: 523, column: 1, scope: !416)
!425 = distinct !DISubprogram(name: "report", linkageName: "_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17hccf942a05a23072aE", scope: !426, file: !78, line: 2432, type: !427, scopeLine: 2432, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, retainedNodes: !429)
!426 = !DINamespace(name: "{impl#57}", scope: !80)
!427 = !DISubroutineType(types: !428)
!428 = !{!79, !7}
!429 = !{!430}
!430 = !DILocalVariable(arg: 1, scope: !425, file: !78, line: 2432, type: !7)
!431 = !DILocation(line: 2432, column: 15, scope: !425)
!432 = !DILocation(line: 2434, column: 6, scope: !425)
!433 = distinct !DISubprogram(name: "with_tail", linkageName: "_ZN14tail_call_test9with_tail17hc3aece45eddbb180E", scope: !435, file: !434, line: 5, type: !436, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, retainedNodes: !438)
!434 = !DIFile(filename: "tail-call-test.rs", directory: "/Users/mhornicky/rust/tests/run-make/tail-call-llvm-ir", checksumkind: CSK_MD5, checksum: "c9d420132c078a5cd245c6d5c9374767")
!435 = !DINamespace(name: "tail_call_test", scope: null)
!436 = !DISubroutineType(types: !437)
!437 = !{!178, !178}
!438 = !{!439}
!439 = !DILocalVariable(name: "n", arg: 1, scope: !433, file: !434, line: 5, type: !178)
!440 = !DILocation(line: 5, column: 18, scope: !433)
!441 = !DILocation(line: 6, column: 8, scope: !433)
!442 = !DILocation(line: 11, column: 2, scope: !433)
!443 = !DILocation(line: 9, column: 26, scope: !433)
!444 = !DILocation(line: 9, column: 9, scope: !433)
!445 = distinct !DISubprogram(name: "no_tail", linkageName: "_ZN14tail_call_test7no_tail17h8d6adcb9d9b56776E", scope: !435, file: !434, line: 15, type: !436, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, retainedNodes: !446)
!446 = !{!447}
!447 = !DILocalVariable(name: "n", arg: 1, scope: !445, file: !434, line: 15, type: !178)
!448 = !DILocation(line: 15, column: 16, scope: !445)
!449 = !DILocation(line: 16, column: 8, scope: !445)
!450 = !DILocation(line: 17, column: 9, scope: !445)
!451 = !DILocation(line: 16, column: 5, scope: !445)
!452 = !DILocation(line: 19, column: 17, scope: !445)
!453 = !DILocation(line: 21, column: 2, scope: !445)
!454 = !DILocation(line: 19, column: 9, scope: !445)
!455 = distinct !DISubprogram(name: "even_with_tail", linkageName: "_ZN14tail_call_test14even_with_tail17h8d1dbeb6515757cfE", scope: !435, file: !434, line: 25, type: !456, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, retainedNodes: !458)
!456 = !DISubroutineType(types: !457)
!457 = !{!238, !178}
!458 = !{!459}
!459 = !DILocalVariable(name: "n", arg: 1, scope: !455, file: !434, line: 25, type: !178)
!460 = !DILocation(line: 25, column: 23, scope: !455)
!461 = !DILocation(line: 26, column: 5, scope: !455)
!462 = !DILocation(line: 30, column: 2, scope: !455)
!463 = !DILocation(line: 28, column: 35, scope: !455)
!464 = !DILocation(line: 28, column: 14, scope: !455)
!465 = distinct !DISubprogram(name: "odd_with_tail", linkageName: "_ZN14tail_call_test13odd_with_tail17hc95516150aaddcc1E", scope: !435, file: !434, line: 32, type: !456, scopeLine: 32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, retainedNodes: !466)
!466 = !{!467}
!467 = !DILocalVariable(name: "n", arg: 1, scope: !465, file: !434, line: 32, type: !178)
!468 = !DILocation(line: 32, column: 22, scope: !465)
!469 = !DILocation(line: 33, column: 5, scope: !465)
!470 = !DILocation(line: 37, column: 2, scope: !465)
!471 = !DILocation(line: 35, column: 36, scope: !465)
!472 = !DILocation(line: 35, column: 14, scope: !465)
!473 = distinct !DISubprogram(name: "even_no_tail", linkageName: "_ZN14tail_call_test12even_no_tail17hd3be8e5729868b97E", scope: !435, file: !434, line: 41, type: !456, scopeLine: 41, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, retainedNodes: !474)
!474 = !{!475}
!475 = !DILocalVariable(name: "n", arg: 1, scope: !473, file: !434, line: 41, type: !178)
!476 = !DILocation(line: 41, column: 21, scope: !473)
!477 = !DILocation(line: 42, column: 5, scope: !473)
!478 = !DILocation(line: 43, column: 14, scope: !473)
!479 = !DILocation(line: 44, column: 26, scope: !473)
!480 = !DILocation(line: 46, column: 2, scope: !473)
!481 = !DILocation(line: 44, column: 14, scope: !473)
!482 = distinct !DISubprogram(name: "odd_no_tail", linkageName: "_ZN14tail_call_test11odd_no_tail17h1399b4ae4de50274E", scope: !435, file: !434, line: 48, type: !456, scopeLine: 48, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !29, templateParams: !23, retainedNodes: !483)
!483 = !{!484}
!484 = !DILocalVariable(name: "n", arg: 1, scope: !482, file: !434, line: 48, type: !178)
!485 = !DILocation(line: 48, column: 20, scope: !482)
!486 = !DILocation(line: 49, column: 5, scope: !482)
!487 = !DILocation(line: 50, column: 14, scope: !482)
!488 = !DILocation(line: 51, column: 27, scope: !482)
!489 = !DILocation(line: 53, column: 2, scope: !482)
!490 = !DILocation(line: 51, column: 14, scope: !482)
!491 = distinct !DISubprogram(name: "main", linkageName: "_ZN14tail_call_test4main17h34b192e87b023961E", scope: !435, file: !434, line: 55, type: !21, scopeLine: 55, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagMainSubprogram, unit: !29, templateParams: !23, retainedNodes: !492)
!492 = !{!493, !495, !497, !499, !501, !503}
!493 = !DILocalVariable(name: "with_tail_result", scope: !494, file: !434, line: 57, type: !178, align: 32)
!494 = distinct !DILexicalBlock(scope: !491, file: !434, line: 57, column: 5)
!495 = !DILocalVariable(name: "no_tail_result", scope: !496, file: !434, line: 58, type: !178, align: 32)
!496 = distinct !DILexicalBlock(scope: !494, file: !434, line: 58, column: 5)
!497 = !DILocalVariable(name: "even_with_tail_result", scope: !498, file: !434, line: 59, type: !238, align: 8)
!498 = distinct !DILexicalBlock(scope: !496, file: !434, line: 59, column: 5)
!499 = !DILocalVariable(name: "odd_with_tail_result", scope: !500, file: !434, line: 60, type: !238, align: 8)
!500 = distinct !DILexicalBlock(scope: !498, file: !434, line: 60, column: 5)
!501 = !DILocalVariable(name: "even_no_tail_result", scope: !502, file: !434, line: 61, type: !238, align: 8)
!502 = distinct !DILexicalBlock(scope: !500, file: !434, line: 61, column: 5)
!503 = !DILocalVariable(name: "odd_no_tail_result", scope: !504, file: !434, line: 62, type: !238, align: 8)
!504 = distinct !DILexicalBlock(scope: !502, file: !434, line: 62, column: 5)
!505 = !DILocation(line: 57, column: 9, scope: !494)
!506 = !DILocation(line: 58, column: 9, scope: !496)
!507 = !DILocation(line: 59, column: 9, scope: !498)
!508 = !DILocation(line: 60, column: 9, scope: !500)
!509 = !DILocation(line: 61, column: 9, scope: !502)
!510 = !DILocation(line: 62, column: 9, scope: !504)
!511 = !DILocation(line: 57, column: 28, scope: !491)
!512 = !DILocation(line: 58, column: 26, scope: !494)
!513 = !DILocation(line: 59, column: 33, scope: !496)
!514 = !DILocation(line: 60, column: 32, scope: !498)
!515 = !DILocation(line: 61, column: 31, scope: !500)
!516 = !DILocation(line: 62, column: 30, scope: !502)
!517 = !DILocation(line: 64, column: 5, scope: !504)
!518 = !DILocation(line: 71, column: 2, scope: !519)
!519 = !DILexicalBlockFile(scope: !491, file: !434, discriminator: 0)
