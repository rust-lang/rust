// checks case typos with libstd::alloc structs

fn test_layout(_x: LayOut){}
//~^ ERROR: cannot find type `LayOut` in this scope
fn test_system(_x: system){}
//~^ ERROR: cannot find type `system` in this scope

// checks case typos with libstd::any structs

fn test_typeid(_x: Typeid){}
//~^ ERROR: cannot find type `Typeid` in this scope

// checks case typos with libstd::ascii structs

fn test_escapedefault(_x: Escapedefault){}
//~^ ERROR: cannot find type `Escapedefault` in this scope

// checks case typos with libstd::cell structs

fn test_cell(_x: cell<()>){}
//~^ ERROR: cannot find type `cell` in this scope

// checks case typos with libstd::char structs

fn test_decodeutf16(_x: DecodeUTF16<()>){}
//~^ ERROR: cannot find type `DecodeUTF16` in this scope

fn test_escapeunicode(_x: Escapeunicode){}
//~^ ERROR: cannot find type `Escapeunicode` in this scope

fn test_tolowercase(_x: Tolowercase){}
//~^ ERROR: cannot find type `Tolowercase` in this scope

fn test_touppercase(_x: Touppercase){}
//~^ ERROR: cannot find type `Touppercase` in this scope

// checks case typos with libstd::cmp structs

fn test_reverse(_x: reverse<()>){}
//~^ ERROR: cannot find type `reverse` in this scope

// checks case typos with libstd::collections structs

fn test_btreemap(_x: BtreeMap<(), ()>){}
//~^ ERROR: cannot find type `BtreeMap` in this scope
fn test_btreeset(_x: BtreeSet<()>){}
//~^ ERROR: cannot find type `BtreeSet` in this scope
fn test_binaryheap(_x: Binaryheap<()>){}
//~^ ERROR: cannot find type `Binaryheap` in this scope
fn test_hashmap(_x: Hashmap<String, ()>){}
//~^ ERROR: cannot find type `Hashmap` in this scope
fn test_hashset(_x: Hashset<()>){}
//~^ ERROR: cannot find type `Hashset` in this scope
fn test_linkedlist(_x: Linkedlist<()>){}
//~^ ERROR: cannot find type `Linkedlist` in this scope
fn test_vecdeque(_x: Vecdeque<()>){}
//~^ ERROR: cannot find type `Vecdeque` in this scope

// checks case typos with libstd::env structs

fn test_args(_x: args){}
//~^ ERROR: cannot find type `args` in this scope
fn test_argsos(_x: Argsos){}
//~^ ERROR: cannot find type `Argsos` in this scope
fn test_splitpaths(_x: Splitpaths<'_>){}
//~^ ERROR: cannot find type `Splitpaths` in this scope
fn test_vars(_x: vars){}
//~^ ERROR: cannot find type `vars` in this scope
fn test_varsos(_x: Varsos){}
//~^ ERROR: cannot find type `Varsos` in this scope

// checks case typos with libstd::ffi structs

fn test_cstr(_x: cStr){}
//~^ ERROR: cannot find type `cStr` in this scope
fn test_osstr(_x: Osstr){}
//~^ ERROR: cannot find type `Osstr` in this scope
fn test_osstring(_x: Osstring){}
//~^ ERROR: cannot find type `Osstring` in this scope

// checks case typos with libstd::fmt structs

fn test_debuglist(_x: Debuglist){}
//~^ ERROR: cannot find type `Debuglist` in this scope
fn test_debugmap(_x: Debugmap){}
//~^ ERROR: cannot find type `Debugmap` in this scope
fn test_debugset(_x: Debugset){}
//~^ ERROR: cannot find type `Debugset` in this scope
fn test_debugstruct(_x: Debugstruct){}
//~^ ERROR: cannot find type `Debugstruct` in this scope
fn test_debugtuple(_x: Debugtuple){}
//~^ ERROR: cannot find type `Debugtuple` in this scope
fn test_fmter(mut _x: formatter){}
//~^ ERROR: cannot find type `formatter` in this scope

// checks case typos with libstd::fs structs

fn test_dirbuilder(_x: Dirbuilder){}
//~^ ERROR: cannot find type `Dirbuilder` in this scope
fn test_direntry(_x: Direntry){}
//~^ ERROR: cannot find type `Direntry` in this scope
fn test_filetype(_x: Filetype){}
//~^ ERROR: cannot find type `Filetype` in this scope
fn test_metadata(_x: MetaData){}
//~^ ERROR: cannot find type `MetaData` in this scope
fn test_openoptions(_x: Openoptions){}
//~^ ERROR: cannot find type `Openoptions` in this scope
fn test_permissions(_x: permissions){}
//~^ ERROR: cannot find type `permissions` in this scope
fn test_readdir(_x: Readdir){}
//~^ ERROR: cannot find type `Readdir` in this scope

// checks case typos with libstd::hash structs

fn test_buildhasherdefault(_x: BuildhasherDefault){}
//~^ ERROR: cannot find type `BuildhasherDefault` in this scope

// checks case typos with libstd::io structs

fn test_bufreader(_x: Bufreader<()>){}
//~^ ERROR: cannot find type `Bufreader` in this scope
fn test_bufwriter(_x: Bufwriter<()>){}
//~^ ERROR: cannot find type `Bufwriter` in this scope
fn test_bytes(_x: bytes<()>){}
//~^ ERROR: cannot find type `bytes` in this scope
fn test_chain(_x: chain<(), ()>){}
//~^ ERROR: cannot find type `chain` in this scope
fn test_cursor(_x: cursor<()>){}
//~^ ERROR: cannot find type `cursor` in this scope
fn test_empty(_x: empty){}
//~^ ERROR: cannot find type `empty` in this scope
fn test_ioslice(_x: Ioslice){}
//~^ ERROR: cannot find type `Ioslice` in this scope
fn test_ioslicemut(_x: IosliceMut){}
//~^ ERROR: cannot find type `IosliceMut` in this scope
fn test_linewriter(_x: Linewriter<()>){}
//~^ ERROR: cannot find type `Linewriter` in this scope
fn test_lines(_x: lines<()>){}
//~^ ERROR: cannot find type `lines` in this scope
fn test_repeat(_x: repeat){}
//~^ ERROR: cannot find type `repeat` in this scope
fn test_sink(_x: sink){}
//~^ ERROR: cannot find type `sink` in this scope
fn test_split(_x: split<()>){}
//~^ ERROR: cannot find type `split` in this scope
fn test_stderr(_x: StdErr){}
//~^ ERROR: cannot find type `StdErr` in this scope
fn test_stderrlock(_x: StdErrLock){}
//~^ ERROR: cannot find type `StdErrLock` in this scope
fn test_stdin(_x: StdIn){}
//~^ ERROR: cannot find type `StdIn` in this scope
fn test_stdinlock(_x: StdInLock){}
//~^ ERROR: cannot find type `StdInLock` in this scope
fn test_stdout(_x: StdOut){}
//~^ ERROR: cannot find type `StdOut` in this scope
fn test_stdoutlock(_x: StdOutLock){}
//~^ ERROR: cannot find type `StdOutLock` in this scope
fn test_take(_x: take){}
//~^ ERROR: cannot find type `take` in this scope

// checks case typos with libstd::iter structs

fn test_cloned(_x: cloned<(), ()>){}
//~^ ERROR: cannot find type `cloned` in this scope
fn test_copied(_x: copied<(), ()>){}
//~^ ERROR: cannot find type `copied` in this scope
fn test_cycle(_x: cycle<(), ()>){}
//~^ ERROR: cannot find type `cycle` in this scope
fn test_enumerate(_x: enumerate<(), ()>){}
//~^ ERROR: cannot find type `enumerate` in this scope
fn test_filter(_x: filter<(), ()>){}
//~^ ERROR: cannot find type `filter` in this scope
fn test_filtermap(_x: Filtermap<(), ()>){}
//~^ ERROR: cannot find type `Filtermap` in this scope
fn test_flatten(_x: flatten<()>){}
//~^ ERROR: cannot find type `flatten` in this scope
fn test_fromfn(_x: Fromfn<()>){}
//~^ ERROR: cannot find type `Fromfn` in this scope
fn test_fuse(_x: fuse<()>){}
//~^ ERROR: cannot find type `fuse` in this scope
fn test_inspect(_x: inspect<(), ()>){}
//~^ ERROR: cannot find type `inspect` in this scope
fn test_map(_x: map<(), ()>){}
//~^ ERROR: cannot find type `map` in this scope
fn test_once(_x: once<()>){}
//~^ ERROR: cannot find type `once` in this scope
fn test_oncewith(_x: Oncewith<()>){}
//~^ ERROR: cannot find type `Oncewith` in this scope
fn test_peekable(_x: peekable<()>){}
//~^ ERROR: cannot find type `peekable` in this scope
fn test_repeatwith(_x: Repeatwith<()>){}
//~^ ERROR: cannot find type `Repeatwith` in this scope
fn test_rev(_x: rev<()>){}
//~^ ERROR: cannot find type `rev` in this scope
fn test_scan(_x: scan<(), (), ()>){}
//~^ ERROR: cannot find type `scan` in this scope
fn test_skip(_x: skip<()>){}
//~^ ERROR: cannot find type `skip` in this scope
fn test_skipwhile(_x: Skipwhile<(), ()>){}
//~^ ERROR: cannot find type `Skipwhile` in this scope
fn test_stepby(_x: Stepby<()>){}
//~^ ERROR: cannot find type `Stepby` in this scope
fn test_successors(_x: successors<()>){}
//~^ ERROR: cannot find type `successors` in this scope
fn test_takewhile(_x: Takewhile<(), ()>){}
//~^ ERROR: cannot find type `Takewhile` in this scope
fn test_zip(_x: zip<(), ()>){}
//~^ ERROR: cannot find type `zip` in this scope

// checks case typos with libstd::marker structs

fn test_phantomdata(_x: Phantomdata){}
//~^ ERROR: cannot find type `Phantomdata` in this scope
fn test_phantompinned(_x: Phantompinned){}
//~^ ERROR: cannot find type `Phantompinned` in this scope

// checks case typos with libstd::mem structs

fn test_discriminant(_x: discriminant<()>){}
//~^ ERROR: cannot find type `discriminant` in this scope
fn test_manuallydrop(_x: Manuallydrop<()>){}
//~^ ERROR: cannot find type `Manuallydrop` in this scope

// checks case typos with libstd::net structs

fn test_incoming(_x: incoming){}
//~^ ERROR: cannot find type `incoming` in this scope
fn test_ipv4addr(_x: IPv4Addr){}
//~^ ERROR: cannot find type `IPv4Addr` in this scope
fn test_ipv6addr(_x: IPv6Addr){}
//~^ ERROR: cannot find type `IPv6Addr` in this scope
fn test_socketaddrv4(_x: SocketAddrv4){}
//~^ ERROR: cannot find type `SocketAddrv4` in this scope
fn test_socketaddrv6(_x: SocketAddrv6){}
//~^ ERROR: cannot find type `SocketAddrv6` in this scope
fn test_tcplistener(_x: TCPListener){}
//~^ ERROR: cannot find type `TCPListener` in this scope
fn test_tcpstream(_x: TCPStream){}
//~^ ERROR: cannot find type `TCPStream` in this scope
fn test_udpsocket(_x: UDPSocket){}
//~^ ERROR: cannot find type `UDPSocket` in this scope

// checks case typos with libstd::num structs

fn test_nonzeroi8(_x: NonZeroi8){}
//~^ ERROR: cannot find type `NonZeroi8` in this scope
fn test_nonzeroi16(_x: NonZeroi16){}
//~^ ERROR: cannot find type `NonZeroi16` in this scope
fn test_nonzeroi32(_x: NonZeroi32){}
//~^ ERROR: cannot find type `NonZeroi32` in this scope
fn test_nonzeroi64(_x: NonZeroi64){}
//~^ ERROR: cannot find type `NonZeroi64` in this scope
fn test_nonzeroi128(_x: NonZeroi128){}
//~^ ERROR: cannot find type `NonZeroi128` in this scope
fn test_nonzerou8(_x: NonZerou8){}
//~^ ERROR: cannot find type `NonZerou8` in this scope
fn test_nonzerou16(_x: NonZerou16){}
//~^ ERROR: cannot find type `NonZerou16` in this scope
fn test_nonzerou32(_x: NonZerou32){}
//~^ ERROR: cannot find type `NonZerou32` in this scope
fn test_nonzerou64(_x: NonZerou64){}
//~^ ERROR: cannot find type `NonZerou64` in this scope
fn test_nonzerou128(_x: NonZerou128){}
//~^ ERROR: cannot find type `NonZerou128` in this scope
fn test_nonzerousize(_x: NonzeroUsize){}
//~^ ERROR: cannot find type `NonzeroUsize` in this scope
fn test_wrapping(_x: wrapping){}
//~^ ERROR: cannot find type `wrapping` in this scope

// checks case typos with libstd::ops structs

fn test_range(_x: range<()>){}
//~^ ERROR: cannot find type `range` in this scope
fn test_rangefrom(_x: Rangefrom<()>){}
//~^ ERROR: cannot find type `Rangefrom` in this scope
fn test_rangefull(_x: Rangefull<()>){}
//~^ ERROR: cannot find type `Rangefull` in this scope
fn test_rangeinclusive(_x: Rangeinclusive<()>){}
//~^ ERROR: cannot find type `Rangeinclusive` in this scope
fn test_rangeto(_x: Rangeto<()>){}
//~^ ERROR: cannot find type `Rangeto` in this scope
fn test_rangetoinclusive(_x: RangetoInclusive<()>){}
//~^ ERROR: cannot find type `RangetoInclusive` in this scope

// checks case typos with libstd::panic structs

fn test_assertunwindsafe(_x: AssertUnwindsafe<()>){}
//~^ ERROR: cannot find type `AssertUnwindsafe` in this scope
fn test_location(_x: location<()>){}
//~^ ERROR: cannot find type `location` in this scope
fn test_panicinfo(_x: Panicinfo<()>){}
//~^ ERROR: cannot find type `Panicinfo` in this scope

// checks case typos with libstd::path structs

fn test_ancestors(_x: ancestors){}
//~^ ERROR: cannot find type `ancestors` in this scope
fn test_components(_x: components){}
//~^ ERROR: cannot find type `components` in this scope
fn test_pathbuf(_x: Pathbuf){}
//~^ ERROR: cannot find type `Pathbuf` in this scope
fn test_prefixcomponent(_x: Prefixcomponent){}
//~^ ERROR: cannot find type `Prefixcomponent` in this scope

// checks case typos with libstd::pin structs

fn test_pin(_x: pin<()>){}
//~^ ERROR: cannot find type `pin` in this scope

// checks case typos with libstd::process structs

fn test_child(_x: child){}
//~^ ERROR: cannot find type `child` in this scope
fn test_childstderr(_x: ChildStdErr){}
//~^ ERROR: cannot find type `ChildStdErr` in this scope
fn test_childstdin(_x: ChildStdIn){}
//~^ ERROR: cannot find type `ChildStdIn` in this scope
fn test_childstdout(_x: ChildStdOut){}
//~^ ERROR: cannot find type `ChildStdOut` in this scope
fn test_command(_x: command){}
//~^ ERROR: cannot find type `command` in this scope
fn test_exitstatus(_x: Exitstatus){}
//~^ ERROR: cannot find type `Exitstatus` in this scope
fn test_output(_x: output){}
//~^ ERROR: cannot find type `output` in this scope
fn test_stdio(_x: StdIo){}
//~^ ERROR: cannot find type `StdIo` in this scope

// checks case typos with libstd::ptr structs

fn test_nonnull(_x: Nonnull<()>){}
//~^ ERROR: cannot find type `Nonnull` in this scope

// checks case typos with libstd::rc structs

fn test_rc(_x: rc<()>){}
//~^ ERROR: cannot find type `rc` in this scope
fn test_weak(_x: weak<()>){}
//~^ ERROR: cannot find type `weak` in this scope

// checks case typos with libstd::string structs

fn test_drain(_x: drain){}
//~^ ERROR: cannot find type `drain` in this scope

// checks case typos with libstd::str structs

fn test_charindices(_x: Charindices){}
//~^ ERROR: cannot find type `Charindices` in this scope
fn test_chars(_x: chars){}
//~^ ERROR: cannot find type `chars` in this scope
fn test_encodeutf16(_x: EncodeUTF16){}
//~^ ERROR: cannot find type `EncodeUTF16` in this scope
fn test_matchindices(_x: Matchindices){}
//~^ ERROR: cannot find type `Matchindices` in this scope
fn test_rmatchindices(_x: RmatchIndices){}
//~^ ERROR: cannot find type `RmatchIndices` in this scope
fn test_rmatches(_x: Rmatches){}
//~^ ERROR: cannot find type `Rmatches` in this scope
fn test_rsplit(_x: Rsplit){}
//~^ ERROR: cannot find type `Rsplit` in this scope
fn test_rsplitn(_x: RSplitn){}
//~^ ERROR: cannot find type `RSplitn` in this scope
fn test_rsplitterminator(_x: RsplitTerminator){}
//~^ ERROR: cannot find type `RsplitTerminator` in this scope
fn test_splitasciiwhitespace(_x: SplitASCIIWhitespace){}
//~^ ERROR: cannot find type `SplitASCIIWhitespace` in this scope
fn test_splitn(_x: Splitn){}
//~^ ERROR: cannot find type `Splitn` in this scope
fn test_splitterminator(_x: Splitterminator){}
//~^ ERROR: cannot find type `Splitterminator` in this scope
fn test_splitwhitespace(_x: Splitwhitespace){}
//~^ ERROR: cannot find type `Splitwhitespace` in this scope

// checks case typos with libstd::sync structs

fn test_arc(_x: arc<()>){}
//~^ ERROR: cannot find type `arc` in this scope
fn test_barrier(_x: barrier<()>){}
//~^ ERROR: cannot find type `barrier` in this scope
fn test_barrierwaitresult(_x: BarrierwaitResult<()>){}
//~^ ERROR: cannot find type `BarrierwaitResult` in this scope
fn test_condvar(_x: CondVar<()>){}
//~^ ERROR: cannot find type `CondVar` in this scope
fn test_mutex(_x: mutex<()>){}
//~^ ERROR: cannot find type `mutex` in this scope
fn test_mutexguard(_x: Mutexguard<()>){}
//~^ ERROR: cannot find type `Mutexguard` in this scope
fn test_rwlock(_x: RWlock<()>){}
//~^ ERROR: cannot find type `RWlock` in this scope
fn test_rwlockreadguard(_x: RWlockReadGuard<()>){}
//~^ ERROR: cannot find type `RWlockReadGuard` in this scope
fn test_rwlockwriteguard(_x: RWlockWriteGuard<()>){}
//~^ ERROR: cannot find type `RWlockWriteGuard` in this scope
fn test_waittimeoutresult(_x: WaittimeoutResult<()>){}
//~^ ERROR: cannot find type `WaittimeoutResult` in this scope

// checks case typos with libstd::task structs

fn test_context(_x: context){}
//~^ ERROR: cannot find type `context` in this scope
fn test_rawwaker(_x: Rawwaker){}
//~^ ERROR: cannot find type `Rawwaker` in this scope
fn test_rawwakervtable(_x: RawwakerVTable){}
//~^ ERROR: cannot find type `RawwakerVTable` in this scope
fn test_waker(_x: waker){}
//~^ ERROR: cannot find type `waker` in this scope

// checks case typos with libstd::thread structs

fn test_builder(_x: builder){}
//~^ ERROR: cannot find type `builder` in this scope
fn test_joinhandle(_x: Joinhandle<()>){}
//~^ ERROR: cannot find type `Joinhandle` in this scope
fn test_localkey(_x: Localkey<()>){}
//~^ ERROR: cannot find type `Localkey` in this scope
fn test_thread(_x: thread){}
//~^ ERROR: cannot find type `thread` in this scope
fn test_threadid(_x: ThreadID){}
//~^ ERROR: cannot find type `ThreadID` in this scope

// checks case typos with libstd::time structs

fn test_duration(_x: duration){}
//~^ ERROR: cannot find type `duration` in this scope
fn test_instant(_x: instant){}
//~^ ERROR: cannot find type `instant` in this scope
fn test_systemtime(_x: Systemtime){}
//~^ ERROR: cannot find type `Systemtime` in this scope

fn test_systemtime2(_x: SystemTime){}
//~^ ERROR: cannot find type `SystemTime` in this scope

struct SystemTome{}
mod st{
    struct SystemTame{}
}

fn main(){}
