//! This test pins the behaviors of some `Rc` and `Arc` operations so that changes are noticed.
//! These operations are frequently used, so they should behave as expected.

//@ only-unix
//@ only-x86_64
//@ compile-flags: --crate-type=lib -Z merge-functions=disabled -C opt-level=3 -C llvm-args=-x86-asm-syntax=intel
//@ assembly-output: emit-asm

use std::mem;
use std::rc::{Rc, Weak as RcWeak};
use std::sync::{Arc, Weak as ArcWeak};

// CHECK-LABEL: rc_new:
// CHECK-DAG: mov edi, 24
// CHECK-DAG: mov esi, 8
// CHECK-NEXT: call qword ptr {{.*}}__rust_alloc
// CHECK-NEXT: test rax, rax
// CHECK-NEXT: je [[RC_NEW_ALLOC_FAIL:\..+]]
// CHECK-DAG: mov qword ptr [rax], 1
// CHECK-DAG: mov qword ptr [rax + 8], 1
// CHECK-DAG: mov qword ptr [rax + 16], rbx
// CHECK-DAG: pop rbx
// CHECK: ret
// CHECK-NEXT: [[RC_NEW_ALLOC_FAIL]]:
// CHECK: mov edi, [[#N:]]
// CHECK-NEXT: mov esi, [[#32-N]]
// CHECK-NEXT: call qword ptr {{.*}}handle_alloc_error
#[unsafe(no_mangle)]
pub fn rc_new(x: usize) -> Rc<usize> {
    Rc::new(x)
}

// CHECK-LABEL: rc_deref:
// CHECK: mov rax, qword ptr [rdi]
// CHECK-NEXT: add rax, 16
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub fn rc_deref(rc: &Rc<usize>) -> &usize {
    rc
}

// CHECK-LABEL: rc_increment:
// CHECK: mov rax, qword ptr [rdi]
// CHECK-NEXT: inc qword ptr [rax]
// CHECK-NEXT: je [[RC_INCREMENT_OVERFLOW:\..+]]
// CHECK-NEXT: ret
// CHECK-NEXT: [[RC_INCREMENT_OVERFLOW]]:
// CHECK-NEXT: ud2
#[unsafe(no_mangle)]
pub fn rc_increment(rc: &Rc<usize>) {
    mem::forget(Rc::clone(rc));
}

// CHECK-LABEL: rc_decrement:
// CHECK: push rax
// CHECK: mov qword ptr [rsp], rdi
// CHECK-NEXT: dec qword ptr [rdi]
// CHECK-NEXT: je [[RC_DECREMENT_DROP_SLOW:\..+]]
// CHECK-NEXT: pop rax
// CHECK: ret
// CHECK-NEXT: [[RC_DECREMENT_DROP_SLOW]]:
// CHECK: mov rdi, rsp
// CHECK-NEXT: call qword ptr {{.*}}drop_slow
// CHECK-NEXT: pop rax
// CHECK: ret
#[unsafe(no_mangle)]
pub fn rc_decrement(rc: Rc<usize>) {
    drop(rc);
}

// CHECK-LABEL: rc_downgrade:
// CHECK: mov rax, qword ptr [rdi]
// CHECK-NEXT: inc qword ptr [rax + 8]
// CHECK-NEXT: je [[RC_DOWNGRADE_OVERFLOW:\..+]]
// CHECK: ret
// CHECK-NEXT: [[RC_DOWNGRADE_OVERFLOW]]:
// CHECK-NEXT: ud2
#[unsafe(no_mangle)]
pub fn rc_downgrade(rc: &Rc<usize>) {
    mem::forget(Rc::downgrade(rc));
}

// CHECK-LABEL: rc_upgrade:
// CHECK: mov rax, qword ptr [rdi]
// CHECK-NEXT: cmp rax, -1
// CHECK-NEXT: je [[RC_UPGRADE_RET:\..+]]
// CHECK-NEXT: mov rcx, qword ptr [rax]
// CHECK-NEXT: test rcx, rcx
// CHECK-NEXT: je [[RC_UPGRADE_RET]]
// CHECK-NEXT: inc rcx
// CHECK-NEXT: mov qword ptr [rax], rcx
// CHECK-NEXT: je [[RC_UPGRADE_OVERFLOW:\..+]]
// CHECK-NEXT: [[RC_UPGRADE_RET]]:
// CHECK-NEXT: ret
// CHECK-NEXT: [[RC_UPGRADE_OVERFLOW]]:
// CHECK-NEXT: ud2
#[unsafe(no_mangle)]
pub fn rc_upgrade(weak: &RcWeak<usize>) {
    mem::forget(weak.upgrade());
}

// CHECK-LABEL: arc_new:
// CHECK-DAG: mov edi, 24
// CHECK-DAG: mov esi, 8
// CHECK-NEXT: call qword ptr {{.*}}__rust_alloc
// CHECK-NEXT: test rax, rax
// CHECK-NEXT: je [[ARC_NEW_ALLOC_FAIL:\..+]]
// CHECK-DAG: mov qword ptr [rax], 1
// CHECK-DAG: mov qword ptr [rax + 8], 1
// CHECK-DAG: mov qword ptr [rax + 16], rbx
// CHECK-DAG: pop rbx
// CHECK: ret
// CHECK-NEXT: [[ARC_NEW_ALLOC_FAIL]]:
// CHECK: mov edi, [[#N:]]
// CHECK-NEXT: mov esi, [[#32-N]]
// CHECK-NEXT: call qword ptr {{.*}}handle_alloc_error
#[unsafe(no_mangle)]
pub fn arc_new(x: usize) -> Arc<usize> {
    Arc::new(x)
}

// CHECK-LABEL: arc_deref:
// CHECK: mov rax, qword ptr [rdi]
// CHECK-NEXT: add rax, 16
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub fn arc_deref(arc: &Arc<usize>) -> &usize {
    arc
}

// CHECK-LABEL: arc_increment:
// CHECK: mov rax, qword ptr [rdi]
// CHECK-NEXT: lock inc qword ptr [rax]
// CHECK-NEXT: jle [[ARC_INCREMENT_OVERFLOW:\..+]]
// CHECK-NEXT: ret
// CHECK-NEXT: [[ARC_INCREMENT_OVERFLOW]]:
// CHECK-NEXT: ud2
#[unsafe(no_mangle)]
pub fn arc_increment(arc: &Arc<usize>) {
    mem::forget(Arc::clone(arc));
}

// CHECK-LABEL: arc_decrement:
// CHECK: push rax
// CHECK: mov qword ptr [rsp], rdi
// CHECK-NEXT: lock dec qword ptr [rdi]
// CHECK-NEXT: jne [[ARC_DECREMENT_RET:\..+]]
// CHECK-NEXT: #MEMBARRIER
// CHECK: mov rdi, rsp
// CHECK-NEXT: call qword ptr {{.*}}drop_slow
// CHECK-NEXT: [[ARC_DECREMENT_RET]]:
// CHECK-NEXT: pop rax
// CHECK: ret
#[unsafe(no_mangle)]
pub fn arc_decrement(arc: Arc<usize>) {
    drop(arc);
}

// CHECK-LABEL: arc_downgrade:
// CHECK: push rax
// CHECK: mov rcx, qword ptr [rdi]
// CHECK-NEXT: jmp [[ARC_DOWNGRADE_LOAD_WEAK:\..+]]
// CHECK: [[ARC_DOWNGRADE_PAUSE:\..+]]:
// CHECK-NEXT: pause
// CHECK-NEXT: [[ARC_DOWNGRADE_LOAD_WEAK]]:
// CHECK-NEXT: mov rax, qword ptr [rcx + 8]
// CHECK: [[ARC_DOWNGRADE_RETRY:\..+]]:
// CHECK-NEXT: cmp rax, -1
// CHECK-NEXT: je [[ARC_DOWNGRADE_PAUSE]]
// CHECK-NEXT: test rax, rax
// CHECK-NEXT: js [[ARC_DOWNGRADE_OVERFLOW:\..+]]
// CHECK-NEXT: lea rdx, [rax + 1]
// CHECK-NEXT: lock cmpxchg qword ptr [rcx + 8], rdx
// CHECK-NEXT: jne [[ARC_DOWNGRADE_RETRY]]
// CHECK: pop rax
// CHECK: ret
// CHECK: [[ARC_DOWNGRADE_OVERFLOW]]:
// CHECK: lea rdi, {{.*}}]
// CHECK-NEXT: call qword ptr {{.*}}panic_arc_overflow
#[unsafe(no_mangle)]
pub fn arc_downgrade(arc: &Arc<usize>) {
    mem::forget(Arc::downgrade(arc));
}

// CHECK-LABEL: arc_upgrade:
// CHECK: push rax
// CHECK: mov rcx, qword ptr [rdi]
// CHECK-NEXT: cmp rcx, -1
// CHECK-NEXT: je [[ARC_UPGRADE_RET:\..+]]
// CHECK-NEXT: mov rax, qword ptr [rcx]
// CHECK: [[ARC_UPGRADE_RETRY:\..+]]:
// CHECK-NEXT: test rax, rax
// CHECK-NEXT: je [[ARC_UPGRADE_RET]]
// CHECK-NEXT: js [[ARC_UPGRADE_OVERFLOW:\..+]]
// CHECK-NEXT: lea rdx, [rax + 1]
// CHECK-NEXT: lock cmpxchg qword ptr [rcx], rdx
// CHECK-NEXT: jne [[ARC_UPGRADE_RETRY]]
// CHECK-NEXT: [[ARC_UPGRADE_RET]]:
// CHECK-NEXT: pop rax
// CHECK: ret
// CHECK: [[ARC_UPGRADE_OVERFLOW]]:
// CHECK: lea rdi, {{.*}}]
// CHECK-NEXT: call qword ptr {{.*}}panic_arc_overflow
#[unsafe(no_mangle)]
pub fn arc_upgrade(weak: &ArcWeak<usize>) {
    mem::forget(weak.upgrade());
}
