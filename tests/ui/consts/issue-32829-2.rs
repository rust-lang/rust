const BAD: u32 = {
    {
        5;
        0
    }
};

const BAD2: u32 = {
    {
        invalid();
        //~^ ERROR: cannot call non-const fn `invalid`
        0
    }
};

const BAD3: u32 = {
    {
        valid();
        0
    }
};

static BAD4: u32 = {
    {
        5;
        0
    }
};

static BAD5: u32 = {
    {
        invalid();
        //~^ ERROR: cannot call non-const fn `invalid`
        0
    }
};

static BAD6: u32 = {
    {
        valid();
        0
    }
};

static mut BAD7: u32 = {
    {
        5;
        0
    }
};

static mut BAD8: u32 = {
    {
        invalid();
        //~^ ERROR: cannot call non-const fn `invalid`
        0
    }
};

static mut BAD9: u32 = {
    {
        valid();
        0
    }
};

fn invalid() {}
const fn valid() {}

fn main() {}
