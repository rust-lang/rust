#![allow(non_camel_case_types, dead_code)]
#![no_std]

pub mod nrf52810_pac {
    extern crate cortex_m_rt;
    pub struct Vector { _handler: unsafe extern "C" fn(), }
    extern "C" fn power_clock_2() { }

    #[link_section = ".vector_table.interrupts"]
    #[no_mangle]
    pub static __INTERRUPTS: [Vector; 1] = [ Vector { _handler: power_clock_2 } ];

    mod ficr {
        mod info {
            mod part {
                #[derive(Debug, PartialEq)]struct PARTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
            }
            mod package {
                #[derive(Debug, PartialEq)]struct PACKAGER;
                struct R;
                impl R { }
                impl R { }
                impl R { }
            }
            mod flash {
                #[derive(Debug, PartialEq)]struct FLASHR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod deviceaddrtype {
            #[derive(Debug, PartialEq)]struct DEVICEADDRTYPER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod bprot {
        mod config0 {
            #[derive(Debug, PartialEq)]struct REGION0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION2R;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION3R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION4R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION5R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION6R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION7R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION8R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION9R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION10R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION11R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION12R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION13R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION14R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION15R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION16R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION17R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION18R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION19R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION20R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION21R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION22R;
            impl R { }
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION23R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION24R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION25R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION26R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION27R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION28R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION29R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION30R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct REGION31R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod disableindebug {
            struct R;
            impl R { }
            #[derive(Clone, Copy, Debug, PartialEq)]struct DISABLEINDEBUGR;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod twi0 {
        mod psel {
            mod scl {
                struct R;
                impl R { }
                impl R { }
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                impl R { }

                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod sda {
                struct R;
                impl R { }
                impl R { }
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod events_stopped {
            #[derive(Debug, PartialEq)]struct EVENTS_STOPPEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_txdsent {
            #[derive(Debug, PartialEq)]struct EVENTS_TXDSENTR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_bb {
            #[derive(Debug, PartialEq)]struct EVENTS_BBR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod shorts {
            #[derive(Debug, PartialEq)]struct BB_SUSPENDR;
            #[derive(Debug, PartialEq)]struct BB_STOPR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenclr {
            #[derive(Debug, PartialEq)]struct STOPPEDR;
            #[derive(Debug, PartialEq)]struct RXDREADYR;
            #[derive(Debug, PartialEq)]struct TXDSENTR;
            #[derive(Debug, PartialEq)]struct ERRORR;
            #[derive(Debug, PartialEq)]struct BBR;
            #[derive(Debug, PartialEq)]struct SUSPENDEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod enable {
            #[derive(Debug, PartialEq)]struct ENABLER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod frequency {
            #[derive(Debug, PartialEq)]struct FREQUENCYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod twim0 {
        mod psel {
            mod scl {
                #[derive(Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod sda {
                #[derive(Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod txd {
            mod ptr {
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod maxcnt {
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod amount {
                struct R;
                impl R { }
                impl R { }
                impl R { }
            }
            mod list {
                #[derive(Debug, PartialEq)]struct LISTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod tasks_suspend {
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_error {
            #[derive(Debug, PartialEq)]struct EVENTS_ERRORR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_rxstarted {
            #[derive(Debug, PartialEq)]struct EVENTS_RXSTARTEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_lastrx {
            #[derive(Debug, PartialEq)]struct EVENTS_LASTRXR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod shorts {
            #[derive(Debug, PartialEq)]struct LASTTX_STARTRXR;
            #[derive(Debug, PartialEq)]struct LASTTX_SUSPENDR;
            #[derive(Debug, PartialEq)]struct LASTTX_STOPR;
            #[derive(Debug, PartialEq)]struct LASTRX_STARTTXR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct LASTRX_SUSPENDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct LASTRX_STOPR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenset {
            #[derive(Clone, Copy, Debug, PartialEq)]struct STOPPEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ERRORR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SUSPENDEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct RXSTARTEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TXSTARTEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct LASTRXR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct LASTTXR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod errorsrc {
            #[derive(Clone, Copy, Debug, PartialEq)]struct OVERRUNR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ANACKR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct DNACKR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod frequency {
            #[derive(Clone, Copy, Debug, PartialEq)]struct FREQUENCYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod twis0 {
        mod psel {
            mod scl {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod sda {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod txd {
            mod ptr{
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod maxcnt {
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod amount {
                struct R;
                impl R { }
                impl R { }
                impl R { }
            }
            mod list {
                #[derive(Clone, Copy, Debug, PartialEq)]struct LISTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod events_stopped {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_STOPPEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_rxstarted {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_RXSTARTEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_write {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_WRITER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod shorts {
            #[derive(Clone, Copy, Debug, PartialEq)]struct WRITE_SUSPENDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct READ_SUSPENDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenset {
            #[derive(Clone, Copy, Debug, PartialEq)]struct STOPPEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ERRORR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct RXSTARTEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TXSTARTEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct WRITER;
            #[derive(Clone, Copy, Debug, PartialEq)]struct READR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod errorsrc {
            #[derive(Debug, PartialEq)]struct OVERFLOWR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct DNACKR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct OVERREADR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod enable {
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENABLER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod config {
            #[derive(Clone, Copy, Debug, PartialEq)]struct ADDRESS0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ADDRESS1R;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod spi0 {
        mod psel {
            mod sck {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }

            }
            mod mosi {
                #[derive(Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod miso {
                #[derive(Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod intenset {
            #[derive(Debug, PartialEq)]struct READYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod enable {
            #[derive(Debug, PartialEq)]struct ENABLER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod config {
            #[derive(Clone, Copy, Debug, PartialEq)]struct ORDERR;
            #[derive(Debug, PartialEq)]struct CPHAR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CPOLR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod spim0 {
        mod psel {
            mod sck {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod mosi {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod miso {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod txd {
            mod ptr {
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod maxcnt {
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod amount {
                struct R;
                impl R { }
                impl R { }
                impl R { }
            }
            mod list {
                #[derive(Clone, Copy, Debug, PartialEq)]struct LISTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod tasks_resume {
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_stopped {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_STOPPEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_end {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_ENDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_started {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_STARTEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenset {
            #[derive(Clone, Copy, Debug, PartialEq)]struct STOPPEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENDRXR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENDTXR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct STARTEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenclr {
            #[derive(Clone, Copy, Debug, PartialEq)]struct STOPPEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENDRXR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENDTXR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct STARTEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod frequency {
            #[derive(Clone, Copy, Debug, PartialEq)]struct FREQUENCYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod spis0 {
        mod psel {
            mod sck {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod miso {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod mosi {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod csn {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod txd {
            mod ptr {
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod maxcnt {
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
            mod amount {
                struct R;
                impl R { }
                impl R { }
                impl R { }
            }
            mod list {
                #[derive(Clone, Copy, Debug, PartialEq)]struct LISTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod events_endrx {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_ENDRXR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod shorts {
            #[derive(Clone, Copy, Debug, PartialEq)]struct END_ACQUIRER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenclr {
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ENDRXR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct ACQUIREDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod status {
            #[derive(Clone, Copy, Debug, PartialEq)]struct OVERREADR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct OVERFLOWR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod config {
            #[derive(Clone, Copy, Debug, PartialEq)]struct ORDERR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CPHAR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CPOLR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    pub struct TIMER0;
    mod timer0 {
        mod shorts {
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE0_CLEARR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE1_CLEARR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE2_CLEARR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE3_CLEARR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE4_CLEARR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE5_CLEARR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE0_STOPR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE1_STOPR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE2_STOPR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE3_STOPR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE4_STOPR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE5_STOPR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenclr {
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE2R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE3R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE4R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct COMPARE5R;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod bitmode {
            #[derive(Clone, Copy, Debug, PartialEq)]struct BITMODER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod temp {
        mod events_datardy {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_DATARDYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenset {
            #[derive(Clone, Copy, Debug, PartialEq)]struct DATARDYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod rng {
        mod events_valrdy {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_VALRDYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenset {
            #[derive(Clone, Copy, Debug, PartialEq)]struct VALRDYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod config {
            #[derive(Clone, Copy, Debug, PartialEq)]struct DERCENR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod egu0 {
        mod inten {
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED2R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED3R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED4R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED5R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED6R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED7R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED8R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED9R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED10R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED11R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED12R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED13R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED14R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED15R;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenclr {
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED2R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED3R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED4R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED5R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED6R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED7R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED8R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED9R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED10R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED11R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED12R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED13R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED14R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct TRIGGERED15R;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod pwm0 {
        mod psel {
            mod out {
                #[derive(Clone, Copy, Debug, PartialEq)]struct CONNECTR;
                struct R;
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
                impl R { }
            }
        }
        mod tasks_seqstart {
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod tasks_nextstep {
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_stopped {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_STOPPEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_seqstarted {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_SEQSTARTEDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_seqend {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_SEQENDR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod events_loopsdone {
            #[derive(Clone, Copy, Debug, PartialEq)]struct EVENTS_LOOPSDONER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod inten {
            #[derive(Clone, Copy, Debug, PartialEq)]struct STOPPEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SEQSTARTED0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SEQSTARTED1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SEQEND0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SEQEND1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct PWMPERIODENDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct LOOPSDONER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod intenclr {
            #[derive(Clone, Copy, Debug, PartialEq)]struct STOPPEDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SEQSTARTED0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SEQSTARTED1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SEQEND0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct SEQEND1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct PWMPERIODENDR;
            #[derive(Clone, Copy, Debug, PartialEq)]struct LOOPSDONER;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod mode {
            #[derive(Clone, Copy, Debug, PartialEq)]struct UPDOWNR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod prescaler {
            #[derive(Clone, Copy, Debug, PartialEq)]struct PRESCALERR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
        mod loop_ {
            #[derive(Clone, Copy, Debug, PartialEq)]struct CNTR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod nvmc {
        mod ready {
            #[derive(Clone, Copy, Debug, PartialEq)]struct READYR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
        }
        mod eraseuicr {
            #[derive(Clone, Copy, Debug, PartialEq)]struct ERASEUICRR;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
    mod ppi {
        mod chg {
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH0R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH1R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH2R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH3R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH4R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH5R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH6R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH7R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH8R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH9R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH10R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH11R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH12R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH13R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH14R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH15R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH16R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH17R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH18R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH19R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH20R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH21R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH22R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH23R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH24R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH25R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH26R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH27R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH28R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH29R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH30R;
            #[derive(Clone, Copy, Debug, PartialEq)]struct CH31R;
            struct R;
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
            impl R { }
        }
    }
}
