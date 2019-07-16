#![allow(non_camel_case_types, unused)]

use libc::{self, c_int, size_t, c_char, BOOL, RTP_DESC, RTP_ID, TASK_ID};


// Copied directly from rtpLibCommon.h, rtpLib.h, signal.h and taskLibCommon.h (for task options)

// ****     definitions for rtpLibCommon.h    ****

pub const RTP_GLOBAL_SYMBOLS     : c_int = 0x01; // register global symbols for RTP
pub const RTP_LOCAL_SYMBOLS      : c_int = 0x02; // idem for local symbols
pub const RTP_ALL_SYMBOLS        : c_int = (RTP_GLOBAL_SYMBOLS | RTP_LOCAL_SYMBOLS);
pub const RTP_DEBUG              : c_int = 0x10; // set RTP in debug mode when created
pub const RTP_BUFFER_VAL_OFF     : c_int = 0x20; // disable buffer validation for all
                                                 // system calls issued from the RTP
pub const RTP_LOADED_WAIT        : c_int = 0x40; // Wait until the RTP is loaded
pub const RTP_CPU_AFFINITY_NONE  : c_int = 0x80; // Remove any CPU affinity (SMP)

// Error Status codes

pub const M_rtpLib : c_int = 178 << 16;

pub const S_rtpLib_INVALID_FILE                   : c_int = (M_rtpLib | 1);
pub const S_rtpLib_INVALID_OPTION                 : c_int = (M_rtpLib | 2);
pub const S_rtpLib_ACCESS_DENIED                  : c_int = (M_rtpLib | 3);
pub const S_rtpLib_INVALID_RTP_ID                 : c_int = (M_rtpLib | 4);
pub const S_rtpLib_NO_SYMBOL_TABLE                : c_int = (M_rtpLib | 5);
pub const S_rtpLib_INVALID_SEGMENT_START_ADDRESS  : c_int = (M_rtpLib | 6);
pub const S_rtpLib_INVALID_SYMBOL_REGISTR_POLICY  : c_int = (M_rtpLib | 7);
pub const S_rtpLib_INSTANTIATE_FAILED             : c_int = (M_rtpLib | 8);
pub const S_rtpLib_INVALID_TASK_OPTION            : c_int = (M_rtpLib | 9);
pub const S_rtpLib_RTP_NAME_LENGTH_EXCEEDED       : c_int = (M_rtpLib | 10);    // rtpInfoGet

pub const VX_RTP_NAME_LENGTH                      : c_int  = 255;    // max name length for diplay


// The 'status' field (32 bit integer) of a RTP holds the RTP state and status.
//
// NOTE: RTP_STATE_GET()    : read the RTP state(s)
//       RTP_STATE_PUT()    : write the RTP state(s)
//       RTP_STATE_SET()    : set a RTP state
//       RTP_STATE_UNSET()  : unset a RTP state
//
//       RTP_STATUS_GET()   : read the RTP status
//       RTP_STATUS_PUT()   : write the RTP status
//       RTP_STATUS_SET()   : set a RTP status
//       RTP_STATUS_UNSET() : unset a RTP status
//
// The PUT/SET/UNSET macros are available only in the kernel headers.


// RTP states

pub const RTP_STATE_CREATE           : c_int  = 0x0001; // RrtpStructTP is under construction
pub const RTP_STATE_NORMAL           : c_int  = 0x0002; // RrtpStructTP is ready
pub const RTP_STATE_DELETE           : c_int  = 0x0004; // RrtpStructTP is being deleted

pub const RTP_STATUS_STOP            : c_int  = 0x0100; // RTP hrtpStructas recieved stopped signal
pub const RTP_STATUS_ELECTED_DELETER : c_int  = 0x0200; // RTP drtpStructelete has started

pub const RTP_STATE_MASK             : c_int  = (RTP_STATE_CREATE | RTP_STATE_NORMAL |
                                                 RTP_STATE_DELETE);
pub const RTP_STATUS_MASK            : c_int  = (RTP_STATUS_STOP | RTP_STATUS_ELECTED_DELETER);

pub fn RTP_STATE_GET  (value : c_int) -> c_int {
    value & RTP_STATE_MASK
}
pub fn RTP_STATUS_GET (value : c_int) -> c_int {
    value & RTP_STATUS_MASK
}

// Indicates that the RTP_ID returned is not valid.

// RTP_ID_ERROR is supposed to be set to -1, but you can't set
// an unsigned value to a negative without casting, and you
// can't cast unless the size of the integer types are the same,
// but the size of RTP_ID may differ between kernel and user space.
// Bitwise or-ing min and max should get the same result.
pub const RTP_ID_ERROR : RTP_ID = RTP_ID::min_value() | RTP_ID::max_value();

// IS_RTP_ C macros

pub fn IS_RTP_STATE_NORMAL           (value : c_int) -> bool {
    (RTP_STATE_GET(value)  & RTP_STATE_NORMAL) == RTP_STATE_NORMAL
}
pub fn IS_RTP_STATE_CREATE           (value : c_int) -> bool {
    (RTP_STATE_GET(value)  & RTP_STATE_CREATE) == RTP_STATE_CREATE
}
pub fn IS_RTP_STATE_DELETE           (value : c_int) -> bool {
    (RTP_STATE_GET(value)  & RTP_STATE_DELETE) == RTP_STATE_DELETE
}
pub fn IS_RTP_STATUS_STOP            (value : c_int) -> bool {
    (RTP_STATUS_GET(value) & RTP_STATUS_STOP ) == RTP_STATUS_STOP
}
pub fn IS_RTP_STATUS_ELECTED_DELETER (value : c_int) -> bool {
    (RTP_STATUS_GET(value) &  RTP_STATUS_ELECTED_DELETER) == RTP_STATUS_ELECTED_DELETER
}

// **** end of definitions for rtpLibCommon.h ****




// ****    definitions for rtpLib.h     ****

pub fn rtpExit(exitCode : c_int) -> ! {
    unsafe{ libc::exit (exitCode) }
}

/* rtpLib.h in the kernel
pub const RTP_DEL_VIA_TASK_DELETE : c_int  = 0x1;          // rtpDelete() via taskDestroy()
pub const RTP_DEL_FORCE           : c_int  = 0x2;          // Forceful  rtpDelete()
pub const RTP_ID_ANY              : RTP_ID = 0;            // used for when a kernel task
                                                           // wants to wait for the next
                                                           // RTP to finish


// Function pointers

pub type RTP_PRE_CREATE_HOOK    = size_t;
pub type RTP_POST_CREATE_HOOK   = size_t;
pub type RTP_INIT_COMPLETE_HOOK = size_t;
pub type RTP_DELETE_HOOK        = size_t;
*/

// **** end of definitions for rtpLib.h ****





// ****     definitions for signal.h    ****
pub fn rtpKill(rtpId : RTP_ID, signo : c_int) -> c_int {
    unsafe{ libc::kill(rtpId as c_int, signo) }
}

pub fn rtpSigqueue(rtpId : RTP_ID, signo : c_int, value : size_t) -> c_int {
    unsafe{ libc::sigqueue(rtpId as c_int, signo, value) }
}

pub fn _rtpSigqueue(rtpId : RTP_ID, signo : c_int, value : *mut size_t, code : c_int) -> c_int {
    unsafe{ libc::_sigqueue(rtpId, signo, value, code) }
}

pub fn taskRaise(signo : c_int) -> c_int {
    unsafe{ libc::taskKill(libc::taskIdSelf(), signo) }
}
pub fn rtpRaise(signo : c_int) -> c_int {
    unsafe{ libc::raise(signo) }
}

// **** end of definitions for signal.h ****



// ****     definitions for taskLibCommon.h    ****
pub const VX_PRIVATE_ENV      : c_int = 0x0080;  // 1 = private environment variables
pub const VX_NO_STACK_FILL    : c_int = 0x0100;  // 1 = avoid stack fill of 0xee
pub const VX_PRIVATE_UMASK    : c_int = 0x0400;  // 1 = private file creation mode mask
pub const VX_TASK_NOACTIVATE  : c_int = 0x2000;  // taskOpen() does not taskActivate()
pub const VX_NO_STACK_PROTECT : c_int = 0x4000;  // no over/underflow stack protection,
                                                 // stack space remains executable

// define for all valid user task options

pub const VX_USR_TASK_OPTIONS_BASE: c_int = (VX_PRIVATE_ENV      |
                                             VX_NO_STACK_FILL    |
                                             VX_TASK_NOACTIVATE  |
                                             VX_NO_STACK_PROTECT |
                                             VX_PRIVATE_UMASK);

// **** end of definitions for taskLibCommon.h ****



extern "C" {
// functions in rtpLibCommon.h

// forward declarations
    pub fn rtpSpawn (
        pubrtpFileName : *const c_char,
        argv           : *const *const c_char,
        envp           : *const *const c_char,
        priority       : c_int,
        uStackSize     : size_t,
        options        : c_int,
        taskOptions    : c_int,
    ) -> RTP_ID;

    pub fn rtpInfoGet (
        rtpId     : RTP_ID,
        rtpStruct : *mut RTP_DESC,
    ) -> c_int;

/* functions in rtpLib.h for kernel


    // function declarations

    pub fn rtpDelete (
        id      : RTP_ID,
        options : c_int,
        status  : c_int,
    ) -> c_int;

    pub fn rtpDeleteForce (
        rtpId : RTP_ID
    ) -> c_int;

    pub fn rtpShow (
        rtpNameOrId : *mut c_char,
        level       : c_int,
    ) -> BOOL;

    // RTP signals are always present when RTPs are included.  The public RTP
    // signal APIs are declared here.


    pub fn rtpKill (
        rtpId : RTP_ID,
        signo : c_int,
    ) -> c_int;

    pub fn rtpSigqueue (
        rtpId : RTP_ID,
        signo : c_int,
        value : size_t, // Actual type is const union sigval value,
                        // which is a union of int and void *
    ) -> c_int;

    pub fn rtpTaskKill (
        tid   : TASK_ID,
        signo : c_int,
    ) -> c_int;

    pub fn rtpTaskSigqueue (
        tid   : TASK_ID,
        signo : c_int,
        value : const size_t, // Actual type is const union sigval,
                        // which is a union of int and void *
    ) -> c_int;

    pub fn rtpWait (
        rtpWaitId : RTP_ID,
        timeout   : libc::alloc_jemalloc_Vx_ticks_t,
        pRtpId    : *mut RTP_ID,
        pStatus   : *mut c_int,
    ) -> c_int;

                             // Other public functions


    pub fn rtpPreCreateHookAdd     (
        hook      : RTP_PRE_CREATE_HOOK,
        addToHead : BOOL,
    ) -> c_int;

    pub fn rtpPreCreateHookDelete  (
        hook      : RTP_POST_CREATE_HOOK,
    ) -> c_int;

    pub fn rtpPostCreateHookAdd    (
        hook      : RTP_POST_CREATE_HOOK,
        addToHead : BOOL,
    ) -> c_int;

    pub fn rtpPostCreateHookDelete (
        hook      : RTP_POST_CREATE_HOOK,
    ) -> c_int;

    pub fn rtpInitCompleteHookAdd  (
        hook      : RTP_INIT_COMPLETE_HOOK,
        addToHead : BOOL,
    ) -> c_int;

    pub fn rtpInitCompleteHookDelete (
        hook      : RTP_INIT_COMPLETE_HOOK,
    ) -> c_int;

    pub fn rtpDeleteHookAdd        (
        hook      : RTP_DELETE_HOOK,
        addToHead : BOOL,
    ) -> c_int;

    pub fn rtpDeleteHookDelete     (
        hook      : RTP_DELETE_HOOK,
    ) -> c_int;

    pub fn rtpMemShow              (
        rtpNameOrId : *mut c_char,
        level       : c_int,
    ) -> c_int;

    pub fn rtpHookShow             (

    );
*/
}
