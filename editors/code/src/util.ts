let enabled: boolean = false;

export const log = {
    debug(message?: any, ...optionalParams: any[]): void {
        if (!enabled) return;
        // eslint-disable-next-line no-console
        console.log(message, ...optionalParams);
    },
    error(message?: any, ...optionalParams: any[]): void {
        if (!enabled) return;
        debugger;
        // eslint-disable-next-line no-console
        console.error(message, ...optionalParams);
    },
    setEnabled(yes: boolean): void {
        enabled = yes;
    }
};
