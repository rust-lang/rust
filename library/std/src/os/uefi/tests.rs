mod env {
    use super::super::env::*;

    #[test]
    fn check_init_first() {
        let mut temp = 10;
        let tmp_ptr: *mut u32 = &mut temp;

        let global_data = GlobalData::new();
        assert!(global_data.init(tmp_ptr).is_ok());
    }

    #[test]
    fn check_init_second() {
        let mut temp = 10;
        let tmp_ptr: *mut u32 = &mut temp;

        let global_data = GlobalData::new();
        assert!(global_data.init(tmp_ptr).is_ok());
        assert!(global_data.init(tmp_ptr).is_err());
    }

    #[test]
    fn check_without_init() {
        let mut global_data: GlobalData<usize> = GlobalData::new();
        assert!(global_data.load().is_err());
    }

    #[test]
    fn check_init_null() {
        let global_data: GlobalData<usize> = GlobalData::new();
        assert!(global_data.init(core::ptr::null_mut()).is_err())
    }

    #[test]
    fn multiple_refernece() {
        let mut temp = 10;
        let tmp_ptr: *mut u32 = &mut temp;

        let global_data = GlobalData::new();
        assert!(global_data.init(tmp_ptr).is_ok());

        let ref1 = global_data.load().map_err(|_| "help").unwrap();
        let ref2 = global_data.load().map_err(|_| "help").unwrap();

        assert_eq!(ref1, ref2);
    }
}
