//@ edition: 2021
//@ compile-flags: -Zno-profiler-runtime
//@ compile-flags: -Cinstrument-coverage -Copt-level=0
//@ revisions: LINUX DARWIN WIN

//@ [LINUX] only-linux
//@ [LINUX] filecheck-flags: -DINSTR_PROF_DATA=__llvm_prf_data
//@ [LINUX] filecheck-flags: -DINSTR_PROF_NAME=__llvm_prf_names
//@ [LINUX] filecheck-flags: -DINSTR_PROF_CNTS=__llvm_prf_cnts
//@ [LINUX] filecheck-flags: -DINSTR_PROF_COVMAP=__llvm_covmap
//@ [LINUX] filecheck-flags: -DINSTR_PROF_COVFUN=__llvm_covfun
//@ [LINUX] filecheck-flags: '-DCOMDAT_IF_SUPPORTED=, comdat'

//@ [DARWIN] only-apple
//@ [DARWIN] filecheck-flags: -DINSTR_PROF_DATA=__DATA,__llvm_prf_data,regular,live_support
//@ [DARWIN] filecheck-flags: -DINSTR_PROF_NAME=__DATA,__llvm_prf_names
//@ [DARWIN] filecheck-flags: -DINSTR_PROF_CNTS=__DATA,__llvm_prf_cnts
//@ [DARWIN] filecheck-flags: -DINSTR_PROF_COVMAP=__LLVM_COV,__llvm_covmap
//@ [DARWIN] filecheck-flags: -DINSTR_PROF_COVFUN=__LLVM_COV,__llvm_covfun
//@ [DARWIN] filecheck-flags: -DCOMDAT_IF_SUPPORTED=

//@ [WIN] only-windows
//@ [WIN] filecheck-flags: -DINSTR_PROF_DATA=.lprfd$M
//@ [WIN] filecheck-flags: -DINSTR_PROF_NAME=.lprfn$M
//@ [WIN] filecheck-flags: -DINSTR_PROF_CNTS=.lprfc$M
//@ [WIN] filecheck-flags: -DINSTR_PROF_COVMAP=.lcovmap$M
//@ [WIN] filecheck-flags: -DINSTR_PROF_COVFUN=.lcovfun$M
//@ [WIN] filecheck-flags: '-DCOMDAT_IF_SUPPORTED=, comdat'

// ignore-tidy-linelength

pub fn will_be_called() -> &'static str {
    let val = "called";
    println!("{}", val);
    val
}

pub fn will_not_be_called() -> bool {
    println!("should not have been called");
    false
}

pub fn print<T>(left: &str, value: T, right: &str)
where
    T: std::fmt::Display,
{
    println!("{}{}{}", left, value, right);
}

pub fn wrap_with<F, T>(inner: T, should_wrap: bool, wrapper: F)
where
    F: FnOnce(&T),
{
    if should_wrap {
        wrapper(&inner)
    }
}

fn main() {
    let less = 1;
    let more = 100;

    if less < more {
        wrap_with(will_be_called(), less < more, |inner| print(" ***", inner, "*** "));
        wrap_with(will_be_called(), more < less, |inner| print(" ***", inner, "*** "));
    } else {
        wrap_with(will_not_be_called(), true, |inner| print("wrapped result is: ", inner, ""));
    }
}

// Check for metadata, variables, declarations, and function definitions injected
// into LLVM IR when compiling with -Cinstrument-coverage.

// WIN:          $__llvm_profile_runtime_user = comdat any

// CHECK-DAG:    @__llvm_coverage_mapping = private constant {{.*}}, section "[[INSTR_PROF_COVMAP]]", align 8

// CHECK-DAG:    @__covrec_{{[A-F0-9]+}}u = linkonce_odr hidden constant {{.*}}, section "[[INSTR_PROF_COVFUN]]"[[COMDAT_IF_SUPPORTED]], align 8

// WIN:          @__llvm_profile_runtime = external{{.*}}global i32

// CHECK:        @__profc__R{{[a-zA-Z0-9_]+}}testprog14will_be_called = {{private|internal}} global
// CHECK-SAME:   section "[[INSTR_PROF_CNTS]]"{{.*}}, align 8

// CHECK:        @__profd__R{{[a-zA-Z0-9_]+}}testprog14will_be_called = {{private|internal}} global
// CHECK-SAME:   @__profc__R{{[a-zA-Z0-9_]+}}testprog14will_be_called
// CHECK-SAME:   section "[[INSTR_PROF_DATA]]"{{.*}}, align 8

// CHECK:        @__profc__R{{[a-zA-Z0-9_]+}}testprog4main = {{private|internal}} global
// CHECK-SAME:   section "[[INSTR_PROF_CNTS]]"{{.*}}, align 8

// CHECK:        @__profd__R{{[a-zA-Z0-9_]+}}testprog4main = {{private|internal}} global
// CHECK-SAME:   @__profc__R{{[a-zA-Z0-9_]+}}testprog4main
// CHECK-SAME:   section "[[INSTR_PROF_DATA]]"{{.*}}, align 8

// CHECK:        @__llvm_prf_nm = private constant
// CHECK-SAME:   section "[[INSTR_PROF_NAME]]", align 1

// CHECK:        @llvm.used = appending global
// CHECK-SAME:   @__llvm_coverage_mapping
// CHECK-SAME:   @__llvm_prf_nm
// CHECK-SAME:   section "llvm.metadata"

// CHECK:        define internal { {{.*}} } @_R{{[a-zA-Z0-9_]+}}testprog14will_be_called() unnamed_addr #{{[0-9]+}} {
// CHECK-NEXT:   start:
// CHECK-NOT:    define internal
// CHECK:        atomicrmw add ptr
// CHECK-SAME:   @__profc__R{{[a-zA-Z0-9_]+}}testprog14will_be_called,

// CHECK:        declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #[[LLVM_INSTRPROF_INCREMENT_ATTR:[0-9]+]]

// WIN:          define linkonce_odr hidden i32 @__llvm_profile_runtime_user() #[[LLVM_PROFILE_RUNTIME_USER_ATTR:[0-9]+]] comdat {
// WIN-NEXT:     %1 = load i32, ptr @__llvm_profile_runtime
// WIN-NEXT:     ret i32 %1
// WIN-NEXT:     }

// CHECK:        attributes #[[LLVM_INSTRPROF_INCREMENT_ATTR]] = { nounwind }
// WIN:          attributes #[[LLVM_PROFILE_RUNTIME_USER_ATTR]] = { noinline }
