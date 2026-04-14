#![no_std]
#![no_main]
#![feature(alloc_error_handler)]

mod arch;
pub mod console;
mod framebuffer;
mod mem;
mod requests;
pub mod runtime;
mod theme;

use arch::hcf;
use core::assert;
use framebuffer::Framebuffer;
use kernel::BootRuntime;

use requests::{BASE_REVISION, FRAMEBUFFER_REQUEST};

pub static RUNTIME: arch::CurrentRuntime = arch::create_runtime();

#[unsafe(no_mangle)]
unsafe extern "C" fn kmain() -> ! {
    // Architecture-specific early initialization (e.g., stack mode switching on AArch64)
    unsafe {
        RUNTIME.early_init();
    }

    assert!(BASE_REVISION.is_supported());

    // Initialize architecture-specific paging (HHDM offset, etc.)
    arch::init_paging();

    // Initialize architecture-specific interrupts (VBAR, etc.)
    unsafe {
        arch::init_interrupts();
    }

    indicate_progress();

    kernel::start(&RUNTIME);
}

fn indicate_progress() {
    if let Some(framebuffer_response) = FRAMEBUFFER_REQUEST.get_response() {
        if let Some(framebuffer) = framebuffer_response.framebuffers().next() {
            // Log raw Limine framebuffer params for debug
            let width = framebuffer.width();
            let height = framebuffer.height();
            let pitch = framebuffer.pitch();
            let bpp = framebuffer.bpp();
            let model = if framebuffer.memory_model() == limine::framebuffer::MemoryModel::RGB {
                "RGB"
            } else {
                "Other"
            };
            let bpp_bytes = (bpp as u64 + 7) / 8;
            let stride = if pitch > 0 {
                let min_stride = width.saturating_mul(4);
                if pitch < min_stride {
                    min_stride
                } else {
                    pitch
                }
            } else {
                width.saturating_mul(4)
            };
            kernel::kinfo!(
                "BOOTFB: limine width={} height={} pitch={} bpp={} model={} -> bpp_bytes={} stride={}",
                width,
                height,
                pitch,
                bpp,
                model,
                bpp_bytes,
                stride
            );
            let _display = Framebuffer::new(&framebuffer);

            // Register disable callback (no-op now but keep for compatibility)
            kernel::syscall::handlers::register_console_disable(theme::disable);
        }
    }
}

#[alloc_error_handler]
fn alloc_error_handler(layout: core::alloc::Layout) -> ! {
    unsafe { kernel::logging::force_unlock() };

    // Get current task info for debugging
    let tid = unsafe { kernel::sched::current_tid_current() };

    kernel::kerror!(
        "OOM: allocation of {} bytes (align={}) failed in task {}",
        layout.size(),
        layout.align(),
        tid
    );

    // Log allocator stats if available (use try_lock to avoid deadlock)
    if let Some(heap) = kernel::memory::kheap::kernel_heap().try_lock() {
        let stats = heap.stats();
        kernel::kerror!(
            "OOM: heap stats: pinned={} evictable={} max_req={}",
            stats.total_pinned_bytes,
            stats.total_evictable_bytes,
            stats.largest_alloc_request
        );
    } else {
        kernel::kerror!("OOM: heap lock held, cannot get stats");
    }

    hcf()
}

#[panic_handler]
fn rust_panic(info: &core::panic::PanicInfo) -> ! {
    // unsafe { kernel::logging::init(&RUNTIME) };
    unsafe { kernel::logging::force_unlock() };
    if let Some(location) = info.location() {
        kernel::kerror!(
            "KERNEL PANIC Location: {}:{}:{} Message: {}",
            location.file(),
            location.line(),
            location.column(),
            info.message()
        );
    } else {
        kernel::kerror!("KERNEL PANIC Location: unknown Message: {}", info.message());
    }
    hcf()
}
